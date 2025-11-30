**SynerGene – AstraZeneca–Sanger DREAM Scripts**

This repository contains the final training code for our SynerGene model on two versions of the AstraZeneca-Sanger DREAM drug-combination dataset.

**Files**

- AstraZeneca_OriginalDataset_FinalCode.py
	Uses the original AstraZeneca–Sanger Excel files
	(Astrazeneca_Main.xlsx and related metadata sheets).

	It:

	- Loads drug pairs, cell lines, IC50 values and the original SYNERGY_SCORE. 
	- Builds molecular graphs from SMILES using RDKit and fixed-size atom features. 
	- Encodes each drug with a chosen GNN backbone (GCN, GIN, GAT, or MPNN) plus global attention pooling. 
	- Encodes IC50(A,B) as a short sequence via LSTM + attention, then fuses:
			Drug A embedding + Drug B embedding + cell-line embedding + IC50 sequence embedding. 
	- Trains a regression-only SynerGene model (no classification head) with K-fold cross-validation, optimizing a combination of MSE and Pearson-correlation–based loss and reporting R², RMSE, MAE, Pearson, and Spearman. 


- astrazeneca_enhanceddataset_finalcode.py
	Uses the cleaned / enhanced CSV version of the dataset
	(all_synergy_summary.csv, all_synergy_per_concentration.csv, Drug_info_release.csv). 

	It provides two related pipelines:
	1. Pair-level SynerGene model (main enhanced dataset)
		- Builds a SynergyDataset where each sample contains:
			- Molecular graphs for Drug A and Drug B (from SMILES)
			- A learned cell-line embedding
			- A dose–response sequence over concentration pairs with per-dose ZIP synergy, encoded by bi-LSTM + attention. 
		
		- Predicts both:
			- A scaled continuous ZIP synergy score (regression)
			- A 3-class synergy label (antagonistic / additive / synergistic) using a classification head. 
		- Trains with a joint loss: MSE on regression + weighted cross-entropy on the classes, with class balancing and metric tracking (R², RMSE, MAE, Pearson, Spearman, ACC, BACC, F1, AUC). 
		
	2. Per-dose SynerGene model (dose-level analysis)
		- Uses SynergyPerDoseDataset: each row is a single concentration pair (conc_A, conc_B) with synergy, plus drug graphs and cell line. 
		- Encodes concentrations via a small MLP and fuses them with the two drug graph embeddings and the cell-line embedding in SynerGenePerDoseModel. 
		- Trains a joint regression + classification model at dose level, again logging full regression and classification metrics.
	
	- The enhanced script also includes utilities to save training histories, plot metric curves and bar charts, and build a processed dataset summarizing model predictions for further analysis in the report. 

**How to Use**

Both scripts are designed to be run in Google Colab or a similar Python environment with the required dependencies (PyTorch, RDKit, NumPy, pandas, scikit-learn, SciPy, Matplotlib) installed and Google Drive mounted to the specified paths. 

- Run AstraZeneca_OriginalDataset_FinalCode.py to reproduce results on the original dataset.
  
- Run astrazeneca_enhanceddataset_finalcode.py to reproduce results on the enhanced dataset, including pair-level and dose-level experiments and plots. 

Together, these two scripts implement the same core SynerGene architecture, allowing a direct comparison between the original and enhanced versions of the AstraZeneca–Sanger DREAM dataset.
