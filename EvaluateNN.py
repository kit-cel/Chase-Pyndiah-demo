import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

build_dir = Path(__file__).resolve().parent / "cmake-build-debug"
sys.path.insert(0, str(build_dir))

import bch_simulator

# ---------------------------------------
# Helper: compute binary classification metrics
# ---------------------------------------
def compute_metrics(y_true, y_pred):
    """
    y_true, y_pred: shape (N, 1) or (N,)
    values must be 0/1
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    TP = np.logical_and(y_pred == 1, y_true == 1).sum()
    TN = np.logical_and(y_pred == 0, y_true == 0).sum()
    FP = np.logical_and(y_pred == 1, y_true == 0).sum()
    FN = np.logical_and(y_pred == 0, y_true == 1).sum()

    accuracy = (y_pred == y_true).mean()

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)          # TPR
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    pos_total = (y_true == 1).sum()
    neg_total = (y_true == 0).sum()

    TPR = TP / (pos_total + 1e-8)
    FNR = FN / (pos_total + 1e-8)
    TNR = TN / (neg_total + 1e-8)
    FPR = FP / (neg_total + 1e-8)

    return {
        "TP": int(TP),
        "TN": int(TN),
        "FP": int(FP),
        "FN": int(FN),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "TPR": float(TPR),
        "FNR": float(FNR),
        "TNR": float(TNR),
        "FPR": float(FPR),
    }
class IdentityActivation(nn.Module):
    def forward(self, x):
        return x


class SmallNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = IdentityActivation()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.out_act(x)
        return x


# Load model
load_path = Path(__file__).resolve().parent / "bch_nn_model.pt"
checkpoint = torch.load(load_path, map_location="cpu")

model = SmallNN(
    input_dim=checkpoint["input_dim"],
    hidden_dim=checkpoint["hidden_dim"]
)
model.load_state_dict(checkpoint["model_state_dict"])

# Generate test data
x_np = bch_simulator.generate_batch(
    batch_size=10000,
    EsNo_dB_min=3.3,
    EsNo_dB_max=3.3
).astype(np.float32)

y_np = x_np[:, 0:1]
cols = list(range(1, 7)) + list(range(8, 12))
X_np = x_np[:, cols]

X_t = torch.from_numpy(X_np)
y_t = torch.from_numpy(y_np)

model.eval()
with torch.no_grad():
    probs_nn = model(X_t)   # X_t should be the same test tensor used for NN eval
    preds_nn = (probs_nn >= 0.5).float().cpu().numpy()

y_true_np = y_np.reshape(-1, 1)   # same test labels
metrics_nn = compute_metrics(y_true_np, preds_nn)

target_tpr = metrics_nn["TPR"]

# Evaluate
with torch.no_grad():
    probs = model(X_t)
    preds = (probs >= 0.5).float()

TP = ((preds == 1) & (y_t == 1)).sum().item()
TN = ((preds == 0) & (y_t == 0)).sum().item()
FP = ((preds == 1) & (y_t == 0)).sum().item()
FN = ((preds == 0) & (y_t == 1)).sum().item()


accuracy = (preds == y_t).float().mean().item()
precision = TP / (TP + FP + 1e-8)
recall = TP / (TP + FN + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)

print("===== Evaluation =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

# Optional normalized confusion matrix entries
pos_total = (y_t == 1).sum().item()
neg_total = (y_t == 0).sum().item()

TPR = TP / (pos_total + 1e-8)   # recall / sensitivity
FNR = FN / (pos_total + 1e-8)
TNR = TN / (neg_total + 1e-8)   # specificity
FPR = FP / (neg_total + 1e-8)

print("\n===== Final Evaluation =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Counts -> TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\n===== Normalized Rates =====")
print(f"TPR / Recall     : {TPR:.4f}")
print(f"FNR              : {FNR:.4f}")
print(f"TNR / Specificity: {TNR:.4f}")
print(f"FPR              : {FPR:.4f}")


# ---------------------------------------
# 2. Optimize threshold t for simple rule:
#    predict 1 if (X[:,3] - X[:,2]) < t
#    while matching NN TPR as closely as possible
# ---------------------------------------
score = X_np[:, 3] - X_np[:, 2]   # IMPORTANT: rows are samples

# set score = 100 where X_np[:, 3] == 0
score[X_np[:, 3] == 0] = 100.0
# Candidate thresholds:
# checking thresholds at score values is enough
candidate_t = np.unique(score)

best_t = None
best_metrics = None
best_gap = float("inf")

for t in candidate_t:
    y_pred_simple = (score < t).astype(np.float32).reshape(-1, 1)
    m = compute_metrics(y_t, y_pred_simple)

    gap = abs(m["TPR"] - target_tpr)

    # tie-breaker: if same TPR gap, prefer higher accuracy
    if (gap < best_gap) or (np.isclose(gap, best_gap) and (best_metrics is None or m["accuracy"] > best_metrics["accuracy"])):
        best_gap = gap
        best_t = float(t)
        best_metrics = m

# Optional:
# Sometimes using midpoints between consecutive scores gives slightly cleaner behavior.
# The above usually works well enough because predictions only change when t crosses a score.


# ---------------------------------------
# 3. Print simple-method metrics
# ---------------------------------------
print("\n===== Simple Threshold Method =====")
print(f"Rule      : predict 1 if (X[:,3] - X[:,2]) < t")
print(f"Best t    : {best_t:.10f}")
print(f"NN TPR    : {target_tpr:.6f}")
print(f"Simple TPR: {best_metrics['TPR']:.6f}")
print(f"TPR gap   : {best_gap:.6e}")

print(f"Accuracy : {best_metrics['accuracy']:.6f}")
print(f"TP: {best_metrics['TP']}, TN: {best_metrics['TN']}, FP: {best_metrics['FP']}, FN: {best_metrics['FN']}")
print(f"Precision: {best_metrics['precision']:.6f}")
print(f"Recall   : {best_metrics['recall']:.6f}")
print(f"F1-score : {best_metrics['f1']:.6f}")
print(f"TNR      : {best_metrics['TNR']:.6f}")
print(f"FPR      : {best_metrics['FPR']:.6f}")
print(f"FNR      : {best_metrics['FNR']:.6f}")
