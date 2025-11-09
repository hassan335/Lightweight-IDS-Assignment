# Lightweight Intrusion Detection System (IDS) — CIC-IDS-2018 (Subset)

This repository contains a lightweight Intrusion Detection System built on an (almost) balanced subset of the **CIC-IDS-2018** dataset.

The project focuses on:
- Using a compact, manageable subset that can be trained on a typical machine.
- Applying simple, efficient machine learning models with high detection performance.
- Providing a clear, reproducible, single-script pipeline.

Models implemented:
**Random Forest (with tuning), Logistic Regression, SVM (RBF), Decision Tree, Soft Voting Ensemble, and Stacking.**

---

## 📊 Dataset

**Main dataset:** CIC-IDS-2018  
Link: https://www.unb.ca/cic/datasets/ids-2018.html

This repository does **not** include the full dataset due to size constraints.

You should:

- Download a subset of the CIC-IDS-2018 dataset (e.g. `CICIDS2018.csv`) from this repo insdie Data.
- Install all dependencies:
  ```bash
  pip install -r requirements.txt
  -Run the below command on gitbash after meeting all requirements
   python Code.py --data <path of dataset>.csv


---

## 🧠 Pipeline (`Code/Code.py`)

`Code.py` is a single end-to-end script that:

1. **Loading & Cleaning**
   - Loads the prepared CIC-IDS-2018 subset.
   - Checks for missing values.
   - Ensures `Label` is binary (0 = Benign, 1 = Attack).
   - Drops highly correlated / redundant features (correlation > 0.9) to reduce complexity.

2. **Exploratory Data Analysis (EDA)**
   - Histograms for selected traffic features.
   - Boxplots by label.
   - Correlation heatmap for important features.
   - Class distribution plot.
   - Preliminary Random Forest feature importance.
   - All key visualizations are saved in `results/`.

3. **Preprocessing**
   - Train–test split (80/20) with stratification.
   - Min–Max scaling of features.

4. **Models Evaluated**
   - Random Forest with `GridSearchCV` (tuned baseline).
   - Random Forest (base using best parameters).
   - Logistic Regression.
   - SVM (RBF, `probability=True`).
   - Decision Tree.
   - Soft Voting Ensemble (RF + LR + SVM + DT).
   - Stacking Classifier (RF + SVM → Logistic Regression).

5. **Evaluation & Saved Outputs**
   - For each model:
     - Accuracy, Precision, Recall, F1-score, ROC-AUC.
     - Confusion matrix (saved as PNG).
   - `results/model_metrics_summary.csv`:
     - Side-by-side comparison of all evaluated models.
   - `results/classification_report_best.txt`:
     - Detailed report for the final chosen model (Stacking).

---

## 📁 Repository Structure

```text
Lightweight-IDS-Assignment/
├── Code/
│   └── Code.py                 # main end-to-end pipeline (EDA → models → results)
├── data/
│   └── README.md               # instructions on obtaining and placing the dataset
├── results/
│   ├── eda_hist_selected_features.png
│   ├── eda_box_selected_features_by_label.png
│   ├── eda_corr_selected_features.png
│   ├── eda_class_distribution.png
│   ├── rf_feature_importance.png
│   ├── cm_rf_tuned.png
│   ├── cm_rf_base.png
│   ├── cm_lr_base.png
│   ├── cm_svm_base.png
│   ├── cm_dt_base.png
│   ├── cm_ensemble_soft_voting.png
│   ├── cm_stacking.png
│   ├── model_metrics_summary.csv
│   └── classification_report_best.txt
├── README.md
└── requirements.txt
