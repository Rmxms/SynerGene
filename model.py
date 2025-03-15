import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv

class GINAttenSyn(nn.Module):
    """
    A synergy prediction model using:
      • GINConv for molecular graphs
      • LSTM for node embeddings
      • Attention-based pooling
      • Reduced hidden dimensions (64) to help on smaller data
    """

    def __init__(
        self,
        molecule_channels: int = 78,
        hidden_channels: int = 64,
        middle_channels: int = 64,
        layer_count: int = 2,
        out_channels: int = 2,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.layer_count = layer_count
        self.hidden_channels = hidden_channels

        # ---------------------------
        # 1) GIN layers + BatchNorm
        # ---------------------------
        self.gin_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        input_dim = molecule_channels
        for i in range(layer_count):
            mlp = nn.Sequential(
                nn.Linear(input_dim if i == 0 else hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.gin_layers.append(GINConv(mlp))
            self.norms.append(nn.BatchNorm1d(hidden_channels))

        # ---------------------------
        # 2) LSTM for node embeddings
        # ---------------------------
        self.border_rnn = nn.LSTM(hidden_channels, hidden_channels, batch_first=True)

        # ---------------------------
        # 3) MLPs to reduce cell features
        # ---------------------------
        self.reduction = nn.Sequential(
            nn.Linear(954, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.reduction2 = nn.Sequential(
            nn.Linear(954, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, molecule_channels),  # e.g. 78
            nn.ReLU()
        )

        # ---------------------------
        # 4) Attention modules
        # ---------------------------
        self.pool1 = Attention(hidden_channels, num_heads=4)
        self.pool2 = Attention(hidden_channels, num_heads=4)

        # ---------------------------
        # 5) Final classifier
        # ---------------------------
        # Input = 2*hidden_channels (GIN-pooled) + 2*hidden_channels (LSTM-pooled) + 256 (cell)
        self.final = nn.Sequential(
            nn.Linear(4 * hidden_channels + 256, middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, out_channels),
        )

    def _forward_molecules(self, x, edge_index, states, batch, norm_layer):
        """
        GIN -> BatchNorm -> ReLU(+res) -> LSTM
        """
        h = self.gin_layers_current(x, edge_index)  # shape => [N, hidden_channels]
        h = norm_layer(h)

        # Conditional skip connection
        if x.shape[-1] == h.shape[-1]:
            h = F.relu(h + x)
        else:
            h = F.relu(h)

        # Single-step LSTM
        h_detach = h.detach()
        rnn_out, (hidden_state, cell_state) = self.border_rnn(
            h_detach.unsqueeze(0),
            states
        )
        rnn_out = rnn_out.squeeze(0)  # => [N, hidden_channels]

        return h, rnn_out, (hidden_state, cell_state)

    def forward(self, mol_left, mol_right):
        x1, edge_index1, batch1, cell, mask1 = (
            mol_left.x,
            mol_left.edge_index,
            mol_left.batch,
            mol_left.cell,
            mol_left.mask,
        )
        x2, edge_index2, batch2, mask2 = (
            mol_right.x,
            mol_right.edge_index,
            mol_right.batch,
            mol_right.mask,
        )

        # Reduce cell
        cell = F.normalize(cell, p=2, dim=1)
        cell_256 = self.reduction(cell)   # => [batch_size, 256]
        cell_78 = self.reduction2(cell)   # => [batch_size, molecule_channels]

        batch_size = torch.max(batch1).item() + 1

        # Expand cell_78 for each node
        cell_78_expanded = cell_78.unsqueeze(1).expand(batch_size, 100, -1)
        cell_78_expanded = cell_78_expanded.reshape(-1, cell_78.size(-1))

        # Add cell features to node embeddings
        h_left = x1 + cell_78_expanded
        h_right = x2 + cell_78_expanded

        mask1 = mask1.reshape((batch_size, 100))
        mask2 = mask2.reshape((batch_size, 100))

        left_states, right_states = None, None

        # Pass through each GIN layer
        for i in range(self.layer_count):
            gin_conv = self.gin_layers[i]
            norm_layer = self.norms[i]

            def gin_layers_current(h, edge_index):
                return gin_conv(h, edge_index)

            self.gin_layers_current = gin_layers_current

            h_left, rnn_out_left, left_states = self._forward_molecules(
                h_left, edge_index1, left_states, batch1, norm_layer
            )
            h_right, rnn_out_right, right_states = self._forward_molecules(
                h_right, edge_index2, right_states, batch2, norm_layer
            )

        # Reshape for attention
        rnn_out_left = rnn_out_left.reshape(batch_size, 100, -1)
        rnn_out_right = rnn_out_right.reshape(batch_size, 100, -1)
        h_left = h_left.reshape(batch_size, 100, -1)
        h_right = h_right.reshape(batch_size, 100, -1)

        # Attention pooling on LSTM
        rnn_pooled_left, rnn_pooled_right = self.pool1(rnn_out_left, rnn_out_right, (mask1, mask2))
        # Attention pooling on GIN
        h_pooled_left, h_pooled_right = self.pool2(h_left, h_right, (mask1, mask2))

        # Combine
        shared_graph_level = torch.cat([h_pooled_left, h_pooled_right], dim=1)
        out = torch.cat([shared_graph_level, rnn_pooled_left, rnn_pooled_right, cell_256], dim=1)
        logits = self.final(out)
        return logits

class Attention(nn.Module):
    """
    Same cross-attention module as before.
    """
    def __init__(self, dim, num_heads=4):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.dim_per_head = dim // num_heads

        self.linear_q = nn.Linear(dim, self.dim_per_head * num_heads)
        self.linear_k = nn.Linear(dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(dim, self.dim_per_head * num_heads)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=0.2)

    def attention(self, q1, k1, v1, q2, k2, v2, attn_mask=None):
        a1 = torch.tanh(torch.bmm(k1, q2.transpose(1, 2)))
        a2 = torch.tanh(torch.bmm(k2, q1.transpose(1, 2)))

        if attn_mask is not None:
            mask1, mask2 = attn_mask
            a1 = torch.sum(a1, dim=2).masked_fill(mask1, float("-inf"))
            a2 = torch.sum(a2, dim=2).masked_fill(mask2, float("-inf"))
        else:
            a1 = torch.sum(a1, dim=2)
            a2 = torch.sum(a2, dim=2)

        a1 = torch.softmax(a1, dim=-1).unsqueeze(dim=1)
        a2 = torch.softmax(a2, dim=-1).unsqueeze(dim=1)

        a1 = self.dropout(a1)
        a2 = self.dropout(a2)

        vector1 = torch.bmm(a1, v1).squeeze(1)
        vector2 = torch.bmm(a2, v2).squeeze(1)
        return vector1, vector2

    def forward(self, seq1, seq2, attn_mask=None):
        q1 = torch.relu(self.linear_q(seq1))
        k1 = torch.relu(self.linear_k(seq1))
        v1 = torch.relu(self.linear_v(seq1))

        q2 = torch.relu(self.linear_q(seq2))
        k2 = torch.relu(self.linear_k(seq2))
        v2 = torch.relu(self.linear_v(seq2))

        vector1, vector2 = self.attention(q1, k1, v1, q2, k2, v2, attn_mask)
        vector1 = self.norm(torch.mean(seq1, dim=1) + vector1)
        vector2 = self.norm(torch.mean(seq2, dim=1) + vector2)
        return vector1, vector2
