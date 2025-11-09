#Libraries Declaration Starts
import argparse
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import os
#Libraries Declaration Ends

# making Directory with name of results
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# results/ folder at project root (sibling of Code/)
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

metrics_summary = []  # if not already defined

#Logging results
def log_metrics(name, y_true, y_pred, y_proba):
    metrics_summary.append({
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    })


def save_cm(name, y_true, y_pred, filename):
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=[0, 1])
    disp.ax_.set_title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
    plt.close()







#Uploading DS
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data", "-d",
    required=True,
    help="Path to the CICIDS2018 CSV file"
)
args = parser.parse_args()

df = pd.read_csv(args.data)

# Remove irrelevant columns, such as timestamps or identifiers that do not
# contribute to classification.
df = df.drop(columns=["Timestamp"])
df.head()
df.info()




# Verifying rows with too many missing values (e.g., >30%).

row_missing = df.isnull().sum(axis=1)

# Percentage of missing values per row
row_missing_percent = (row_missing / df.shape[1]) * 100


# Show rows with >30% missing values
rows_with_many_nans = df[row_missing_percent > 30]
print("\n",rows_with_many_nans.shape)

# output (total rows where missing values >30%, total number of columns)



# For numerical features with occasional NaNs, replacing it with mean.cls
num_cols = df.select_dtypes(include="number").columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

#Replace inf or -inf with the column’s maximum/minimum finite value
# numeric columns only
num_cols = df.select_dtypes(include="number").columns
# finite-only stats (ignore inf/-inf and NaN)
finite = df[num_cols].replace([np.inf, -np.inf], np.nan)
col_max = finite.max()  # per-column finite max
col_min = finite.min()  # per-column finite min

# replace +inf with that column's finite max, -inf with finite min
for c in num_cols:
    df.loc[np.isposinf(df[c]), c] = col_max[c]
    df.loc[np.isneginf(df[c]), c] = col_min[c]



#Verifying data integrity if NAN remains or not
na_count = df.isnull().sum()
na_percent = (na_count / len(df)) * 100
na_table = pd.concat([na_count, na_percent], axis=1, keys=['NA_Count', 'NA_%'])

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(na_table)    


# Convert the attack labels to a binary target variable: 0 → Benign 1 → Attack

df['Label'] = df['Label'].apply(lambda x: 0 if x == 'Benign' else 1)



# Count how many 0's and 1's are in your label column
df['Label'].value_counts()


# Encode destination port numbers into binary features based on standard port ranges:

def encode_dst_port(port: int):
    if 0 <= port <= 1023:
        return 1, 0      # WellKnown, Registered
    elif 1024 <= port <= 49151:
        return 0, 1      # WellKnown, Registered
    else:
        return 0, 0      # Dynamic/Private (base)

df["Dest_Port_WellKnown"], df["Dest_Port_Registered"] = zip(*df["Dst Port"].map(encode_dst_port))
df = df.drop(columns=["Dst Port"])

# Map to names; anything else -> OTHER
proto_map = {6: "TCP", 17: "UDP",0: "OTHER"}
df["Protocol"] = df["Protocol"].map(proto_map).fillna("OTHER")

# Force all categories so you always get TCP/UDP/OTHER columns
cat = CategoricalDtype(categories=["TCP", "UDP", "OTHER"])
df["Protocol"] = df["Protocol"].astype(cat)

# One-hot with integer 0/1
dummies = pd.get_dummies(df["Protocol"], dtype="int8").add_prefix("Protocol_ ")
df = df.drop(columns=["Protocol"]).join(dummies)

print()
df.info()

# Histogram Starts
# Set Seaborn style for cleaner plots
sns.set(style="whitegrid")
# --- Selected numeric features to visualize ---
selected_features = [
    "Flow Duration",
    "Tot Fwd Pkts",
    "Fwd Pkt Len Mean",
    "Flow Byts/s"
]




# --- Plot histograms ---
plt.figure(figsize=(9, 7))

for i, feature in enumerate(selected_features, 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[feature], bins=50, kde=True, color="skyblue")
    
    # Use log scale if data is highly skewed
    if df[feature].max() > 10000:
        plt.xscale("log")
    
    plt.title(f"{feature} Distribution")
    plt.xlabel(feature)
    plt.ylabel("Count")

# plt.tight_layout()
# plt.show()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "eda_hist_selected_features.png"), dpi=300)
plt.close()


# Histogram Ends

# #Box Plot Starts

plt.figure(figsize=(9, 7))

for i, feature in enumerate(selected_features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x="Label", y=feature, data=df, showfliers=False)
    if (df[feature] > 0).all() and df[feature].max() > 10000:
        plt.yscale("log")  # handle extreme skew
    plt.title(f"{feature} by Label")
    plt.xlabel("Label")
    plt.ylabel(feature)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "eda_box_selected_features_by_label.png"), dpi=300)
plt.close()


# # Box Plot Ends


# Correlation of Selected Features Starts


plt.figure(figsize=(6,4))
#cols = ["Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "Flow Pkts/s", "Flow Byts/s"]

cols = [
    "Flow Duration",
    "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Fwd Pkt Len Mean", "Bwd Pkt Len Mean",
    "Flow Byts/s", "Flow Pkts/s",
    "Idle Mean"
]
corr = df[cols].corr()

plt.figure(figsize=(12, 8))  # a bit taller
ax = sns.heatmap(
    corr,
    cmap="coolwarm",
    vmin=-1, vmax=1,
    annot=True,
    fmt=".2f",
    annot_kws={"size":6}
)

plt.title("Correlation (selected features)", fontsize=10)
plt.xticks(rotation=45, ha="right", fontsize=7)
plt.yticks(fontsize=7)

# key line: push plot up so x-labels are fully visible
plt.gcf().subplots_adjust(bottom=0.35)

# plt.show()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "eda_corr_selected_features.png"), dpi=300)
plt.close()



# Correlation of Selected Features Starts
TARGET_COL = "Label"
THRESHOLD = 0.9

# Columns you NEVER want to drop
PROTECTED_COLS = [
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Fwd Pkt Len Mean",
    "Bwd Pkt Len Mean",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Idle Mean",
    "Protocol_ UDP",
]

# 1) Only features for correlation (exclude Label)
feature_df = df.drop(columns=[TARGET_COL])

# 2) Correlation matrix (absolute)
corr = feature_df.corr(numeric_only=True).abs()

# 3) Upper triangle to avoid duplicates
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

# 4) Protect only those that actually exist in df
protected = set(df.columns).intersection(PROTECTED_COLS)

# 5) Decide which columns to drop
cols_to_drop = []
for col in upper.columns:
    if col in protected:
        continue  # never drop protected features
    if any(upper[col] > THRESHOLD):
        cols_to_drop.append(col)

print("Dropping highly correlated columns (> {:.2f}):".format(THRESHOLD))
for col in cols_to_drop:
    print(" -", col)

# 6) Drop from original df (Label + protected stay)
df = df.drop(columns=cols_to_drop)


# Correlation of Selected Features ends


 # Class Imbalance Starts

label_counts = df["Label"].value_counts()
print(label_counts)
print((label_counts / len(df)) * 100)

plt.figure(figsize=(6,4))
label_counts.plot(kind="bar")
plt.title("Class distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "eda_class_distribution.png"), dpi=300)
plt.close()


#  Class Imbalance Ends

# Spliting 80/20 for Test & Train Starts



X = df.drop(columns=['Label'])
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# assert "Label" not in X_train.columns
# assert "Label" not in X_test.columns

# Spliting 80/20 for Test & Train Ends

# Data Scaling Starts


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)   # learns from training only
X_test_scaled  = scaler.transform(X_test)        # uses same rules on test

# Data Scaling Ends

#  Training Random Forest on scaled data Starts


X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=X_train.columns)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

importances = rf.feature_importances_
feature_names = X_train_scaled.columns
# rest same as above


idx = np.argsort(importances)[::-1]
top_n = 15

plt.figure(figsize=(8, 6))
sns.barplot(x=importances[idx][:top_n],
            y=feature_names[idx][:top_n],
            orient="h")
plt.title("Random Forest Feature Importance (Preliminary)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "rf_feature_importance.png"), dpi=300)
plt.close()


# Ends Training Random Forest on scaled data 


#Evaluation Metrice Starts

def eval_and_print(model, name, X_test, y_test, save_cm_flag=False, cm_filename=None):
    y_pred = model.predict(X_test)

    # Probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # e.g., if using decision_function
        scores = model.decision_function(X_test)
        y_proba = MinMaxScaler().fit_transform(scores.reshape(-1, 1)).ravel()

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print(f"\n=== {name} — Test Metrics ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
    print("\nConfusion matrix (labels=[0,1]):\n", cm)
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))

    # store in summary table
    log_metrics(name, y_test, y_pred, y_proba)

    # optionally save confusion matrix image
    if save_cm_flag and cm_filename is not None:
        save_cm(name, y_test, y_pred, cm_filename)

    return y_pred, y_proba




#Evaluation Metrice Ends

#Baseline Model RF Starts

# ---- Baseline Random Forest with GridSearchCV ----

rf = RandomForestClassifier(random_state=42, n_jobs=-1)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
}

grid_rf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring="f1",      # or "accuracy" / "f1_macro" depending on your spec
    n_jobs=-1,
    verbose=1
)

grid_rf.fit(X_train_scaled, y_train)

print("Best params (RF baseline):", grid_rf.best_params_)
print("Best CV score:", grid_rf.best_score_)


best_rf = grid_rf.best_estimator_

# Tuned RF (save confusion matrix)
eval_and_print(
    best_rf,
    "Random Forest (Baseline + GridSearchCV)",
    X_test_scaled, y_test,
    save_cm_flag=True,
    cm_filename="cm_rf_tuned.png"
)
# ---- Baseline Random Forest with GridSearchCV Ends ----

# 1) Random Forest (use best tuned model from GridSearch as RF base learner)
rf_base = best_rf
eval_and_print(
    rf_base,
    "Random Forest (Base)",
    X_test_scaled, y_test,
    save_cm_flag=True,
    cm_filename="cm_rf_base.png"
)



# 2) Logistic Regression
lr_base = LogisticRegression(
    max_iter=1000,
    solver="lbfgs",
    n_jobs=-1
)
lr_base.fit(X_train_scaled, y_train)

# Logistic Regression (Base) with confusion matrix saved
y_pred_lr, y_proba_lr = eval_and_print(
    lr_base,
    "Logistic Regression (Base)",
    X_test_scaled, y_test,
    save_cm_flag=True,
    cm_filename="cm_lr_base.png"
)


# 3) Support Vector Machine (enable probability for ROC-AUC + soft voting)
svm_base = SVC(
    kernel="rbf",
    C=1.0,
    probability=True,   # important for ROC-AUC + soft voting
    random_state=42
)
svm_base.fit(X_train_scaled, y_train)

# SVM (RBF, Base) with confusion matrix saved
y_pred_svm, y_proba_svm = eval_and_print(
    svm_base,
    "SVM (RBF, Base)",
    X_test_scaled, y_test,
    save_cm_flag=True,
    cm_filename="cm_svm_base.png"
)


# 4) Decision Tree
dt_base = DecisionTreeClassifier(
    max_depth=None,
    min_samples_split=2,
    random_state=42
)
dt_base.fit(X_train_scaled, y_train)

# Decision Tree (Base) with confusion matrix saved
y_pred_dt, y_proba_dt = eval_and_print(
    dt_base,
    "Decision Tree (Base)",
    X_test_scaled, y_test,
    save_cm_flag=True,
    cm_filename="cm_dt_base.png"
)




#Build a soft-voting ensemble from RF, LR, SVM, and DT (using tuned RF params)
ensemble = VotingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(
            n_estimators=grid_rf.best_params_["n_estimators"],
            max_depth=grid_rf.best_params_["max_depth"],
            min_samples_split=grid_rf.best_params_["min_samples_split"],
            random_state=42,
            n_jobs=-1
        )),
        ("lr", LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            n_jobs=-1
        )),
        ("svm", SVC(
            kernel="rbf",
            C=1.0,
            probability=True,      # needed for soft voting + ROC-AUC
            random_state=42
        )),
        ("dt", DecisionTreeClassifier(
            max_depth=None,
            min_samples_split=2,
            random_state=42
        )),
    ],
    voting="soft",   # average predicted probabilities
    n_jobs=-1
)

# Train ensemble on the same scaled training data
ensemble.fit(X_train_scaled, y_train)

# Evaluate with your existing helper
y_pred_ens, y_proba_ens = eval_and_print(
    ensemble,
    "Ensemble (Soft Voting)",
    X_test_scaled, y_test,
    save_cm_flag=True,
    cm_filename="cm_ensemble_soft_voting.png"
)

#Build a soft-voting ensemble from RF, LR, SVM, and DT (using tuned RF params) ends

# Building and Trainging Stacking Classifier Starts




# Base learners for stacking (fresh models, NOT already-fitted ones)
stack_estimators = [
    ("rf", RandomForestClassifier(
        n_estimators=grid_rf.best_params_["n_estimators"],
        max_depth=grid_rf.best_params_["max_depth"],
        min_samples_split=grid_rf.best_params_["min_samples_split"],
        random_state=42,
        n_jobs=-1
    )),
    ("svm", SVC(
        kernel="rbf",
        C=1.0,
        probability=True,   # required so stacking can use probabilities
        random_state=42
    )),
]

# Meta-learner: Logistic Regression on top of RF + SVM outputs
stack_model = StackingClassifier(
    estimators=stack_estimators,
    final_estimator=LogisticRegression(
        max_iter=1000,
        solver="lbfgs"
    ),
    stack_method="auto",   # uses predict_proba where available
    n_jobs=-1,
    passthrough=False      # only meta-features (RF/SVM outputs) go to LR
)

# Train stacking model
stack_model.fit(X_train_scaled, y_train)

# Evaluate stacking model with your existing helper
# Evaluate stacking model (FINAL MODEL) with confusion matrix saved
y_pred_stack, y_proba_stack = eval_and_print(
    stack_model,
    "Stacking (RF + SVM → LR)",
    X_test_scaled, y_test,
    save_cm_flag=True,
    cm_filename="cm_stacking.png"
)


# Building and Trainging Stacking Classifier Ends
# Save comparison table for all models
# Save comparison table for all models
pd.DataFrame(metrics_summary).to_csv(
    os.path.join(RESULTS_DIR, "model_metrics_summary.csv"),
    index=False
)

# Final model: Stacking
final_report = classification_report(y_test, y_pred_stack, digits=4)
with open(os.path.join(RESULTS_DIR, "classification_report_best.txt"), "w", encoding="utf-8") as f:
    f.write("Stacking (RF + SVM -> LR) - Final Model\n\n")
    f.write(final_report)





