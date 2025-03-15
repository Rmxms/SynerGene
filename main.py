import random
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from model import GINAttenSyn
from utils import MyDataset, collate
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    cohen_kappa_score,
    accuracy_score,
    precision_score,
    balanced_accuracy_score,
    f1_score,
    recall_score
)
from sklearn import metrics
from torch.utils.data import DataLoader
import copy

# Optional: Focal Loss if you have class imbalance
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, weight=self.weight, reduction=self.reduction)
        return loss

# -----------------------
# 1. Hyperparameters
# -----------------------
TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 128
LR = 0.0005
NUM_EPOCHS = 30
EARLY_STOP_PATIENCE = 5
IMPROVEMENT_THRESHOLD = 0.001
WEIGHT_DECAY = 0.01
NUM_FOLDS = 3
ENSEMBLE_SEEDS = [0, 1]  # You can add more seeds if desired

# Logging
LOG_INTERVAL = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# -----------------------
# 2. Load dataset
# -----------------------
dataset = MyDataset(file_path="./data.pt", subset_size=2000)  # Adjust path if needed
length_dataset = len(dataset)
print(f"Dataset length: {length_dataset}")

indices = list(range(length_dataset))
random.shuffle(indices)
fold_size = length_dataset // NUM_FOLDS

# We'll define a function to train a single model
def train_single_model(seed, train_data, val_data):
    print(f"\n=== Training model with seed {seed} ===")
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Build model
    model = GINAttenSyn(
        molecule_channels=78,
        hidden_channels=64,
        middle_channels=64,
        layer_count=2,
        out_channels=2,
        dropout_rate=0.2
    ).to(device)

    # If classes are imbalanced, try focal:
    # loss_fn = FocalLoss(gamma=2.0)
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # We'll use a simple scheduler or none. Feel free to use CosineAnnealing if you prefer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5, verbose=False)

    loader_train = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate)
    loader_val = DataLoader(val_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)

    best_auc = 0.0
    no_improve_count = 0
    best_state = None

    for epoch in range(1, NUM_EPOCHS + 1):
        # 1) Training
        model.train()
        for batch_idx, (data1, data2, y) in enumerate(loader_train):
            data1, data2, y = data1.to(device), data2.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(data1, data2)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

            if batch_idx % LOG_INTERVAL == 0:
                print(f"Epoch {epoch} [{batch_idx * len(data1.x)}/{len(loader_train.dataset)}] Loss: {loss.item():.4f}")

        # 2) Validation
        model.eval()
        val_labels = []
        val_preds = []
        with torch.no_grad():
            for data1, data2, y in loader_val:
                data1, data2 = data1.to(device), data2.to(device)
                out = model(data1, data2)
                probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
                val_preds.extend(probs)
                val_labels.extend(y.numpy())

        auc_val = roc_auc_score(val_labels, val_preds)
        scheduler.step(auc_val)

        if auc_val > best_auc + IMPROVEMENT_THRESHOLD:
            best_auc = auc_val
            no_improve_count = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improve_count += 1
            if no_improve_count >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load the best state
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best AUC for seed {seed}: {best_auc:.4f}")
    return model

def predict_ensemble(models, test_data):
    """Average predictions from multiple models."""
    loader_test = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data1, data2, y in loader_test:
            data1, data2 = data1.to(device), data2.to(device)
            # sum predictions from each model
            ensemble_scores = torch.zeros(len(y), dtype=torch.float32)
            for m in models:
                m.eval()
                out = m(data1, data2)
                probs = F.softmax(out, dim=1)[:, 1].cpu()  # Probability for class=1
                ensemble_scores += probs
            # average
            ensemble_scores /= len(models)
            all_preds.append(ensemble_scores)
            all_labels.append(y)

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    return all_labels, all_preds

fold_metrics = []

for fold_idx in range(NUM_FOLDS):
    val_start = fold_idx * fold_size
    val_end = val_start + fold_size if fold_idx < NUM_FOLDS - 1 else length_dataset
    val_idx = indices[val_start:val_end]
    train_idx = indices[:val_start] + indices[val_end:]

    data_val = dataset.get_data(val_idx)
    data_train = dataset.get_data(train_idx)

    # Train ensemble seeds
    ensemble_models = []
    for seed in ENSEMBLE_SEEDS:
        model_seed = train_single_model(seed, data_train, data_val)
        ensemble_models.append(model_seed)

    # Evaluate on the fold's test portion
    labels, preds = predict_ensemble(ensemble_models, data_val)

    # Metrics
    AUC = roc_auc_score(labels, preds)
    precision_curve, recall_curve, _ = metrics.precision_recall_curve(labels, preds)
    PR_AUC = metrics.auc(recall_curve, precision_curve)
    Y = (preds >= 0.5).astype(int)
    BACC = balanced_accuracy_score(labels, Y)
    tn, fp, fn, tp = confusion_matrix(labels, Y).ravel()
    TPR = tp / (tp + fn) if (tp + fn) else 0
    PREC = precision_score(labels, Y, zero_division=0)
    ACC = accuracy_score(labels, Y)
    KAPPA = cohen_kappa_score(labels, Y)
    recall_val = recall_score(labels, Y, zero_division=0)
    precision_val = precision_score(labels, Y, zero_division=0)
    F1 = f1_score(labels, Y, zero_division=0)

    print(f"\n=== Fold {fold_idx} Results ===")
    print(f"AUC={AUC:.4f}, PR_AUC={PR_AUC:.4f}, ACC={ACC:.4f}, BACC={BACC:.4f}, F1={F1:.4f}")
    fold_metrics.append([AUC, PR_AUC, ACC, BACC, F1])

fold_metrics = np.array(fold_metrics)
mean_metrics = fold_metrics.mean(axis=0)
std_metrics = fold_metrics.std(axis=0)

print("\n=== Final Cross-Validation Metrics (Averaged) ===")
print(f"AUC = {mean_metrics[0]:.4f} ± {std_metrics[0]:.4f}")
print(f"PR_AUC = {mean_metrics[1]:.4f} ± {std_metrics[1]:.4f}")
print(f"ACC = {mean_metrics[2]:.4f} ± {std_metrics[2]:.4f}")
print(f"BACC = {mean_metrics[3]:.4f} ± {std_metrics[3]:.4f}")
print(f"F1 = {mean_metrics[4]:.4f} ± {std_metrics[4]:.4f}")
