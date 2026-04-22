import sys
from pathlib import Path

# Make sure Python can find the compiled pybind module
build_dir = Path(__file__).resolve().parent / "cmake-build-debug"
sys.path.insert(0, str(build_dir))

import bch_simulator
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ----------------------------
# 1. Reproducibility
# ----------------------------
torch.manual_seed(42)
np.random.seed(42)

# ----------------------------
# 2. Generate training data
# ----------------------------
# Each row of x:
# [is_miscorrection, omega, convertRatio, best_five_metrices(5), best_five_distrED(5)]
#
# First value is label, remaining 13 values are features.
num_samples = 5000
x_np = bch_simulator.generate_batch(
    batch_size=10000,
    EsNo_dB_min=3.0,
    EsNo_dB_max=3.1
).astype(np.float32)

y_np = x_np[:, 0:1]      # shape (N, 1), binary labels
cols = list(range(1, 7)) + list(range(8, 12))
X_np = x_np[:, cols]  # shape (N, 10)


# y_true = y_np
# y_pred = ((X_np[:, 3] - X_np[:, 2]) <3.6/256.0).astype(float).reshape(-1, 1)
#
# # Accuracy
# accuracy = (y_pred == y_true).mean().item()
#
# # Confusion matrix components
# TP = ((y_pred == 1) & (y_true == 1)).sum().item()
# TN = ((y_pred == 0) & (y_true == 0)).sum().item()
# FP = ((y_pred == 1) & (y_true == 0)).sum().item()
# FN = ((y_pred == 0) & (y_true == 1)).sum().item()
#
# # Precision / Recall / F1 (safe divide)
# precision = TP / (TP + FP + 1e-8)
# recall = TP / (TP + FN + 1e-8)
# f1 = 2 * precision * recall / (precision + recall + 1e-8)
#
# print("\n===== Final Evaluation =====")
# print(f"Accuracy : {accuracy:.4f}")
# print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall   : {recall:.4f}")
# print(f"F1-score : {f1:.4f}")

print("Raw data shape:", x_np.shape)
print("Feature shape:", X_np.shape)
print("Label shape:", y_np.shape)


# ----------------------------
# 3. Train / validation split
# ----------------------------
num_total = X_np.shape[0]
perm = np.random.permutation(num_total)

train_ratio = 0.8
num_train = int(train_ratio * num_total)

train_idx = perm[:num_train]
val_idx = perm[num_train:]

X_train = X_np[train_idx]
y_train = y_np[train_idx]

X_val = X_np[val_idx]
y_val = y_np[val_idx]


# ----------------------------
# 4. Feature normalization
# ----------------------------
# Strongly recommended because your features are on very different scales.
# mean = X_train.mean(axis=0, keepdims=True)
# std = X_train.std(axis=0, keepdims=True)
# std[std < 1e-8] = 1.0
#
# X_train = (X_train - mean) / std
# X_val = (X_val - mean) / std


# Convert to torch tensors
X_train_t = torch.from_numpy(X_train)
y_train_t = torch.from_numpy(y_train)

X_val_t = torch.from_numpy(X_val)
y_val_t = torch.from_numpy(y_val)


# ----------------------------
# 5. Define the neural network
# ----------------------------
class IdentityActivation(nn.Module):
    def forward(self, x):
        return x


class SmallNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = IdentityActivation()   # linear activation y = x
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.out_act = nn.Sigmoid()
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.out_act(x)
        return x


input_dim = X_train.shape[1]
hidden_dim = 1#doesn't really matter since no non-linearity

model = SmallNN(input_dim=input_dim, hidden_dim=hidden_dim)
print(model)


# ----------------------------
# 6. Loss and optimizer
# ----------------------------
# Suitable loss: Binary Cross Entropy, because output is sigmoid probability and target is 0/1.
criterion = nn.BCELoss()

# Adam is a solid default optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# ----------------------------
# 7. Training loop
# ----------------------------
num_epochs = 5000
batch_size = 1280

train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()

        preds = model(xb)  # shape (B, 1)
        loss = criterion(preds, yb)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)

        pred_labels = (preds >= 0.5).float()
        running_correct += (pred_labels == yb).sum().item()
        running_total += xb.size(0)

    train_loss = running_loss / running_total
    train_acc = running_correct / running_total

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t)
        val_loss = criterion(val_preds, y_val_t).item()
        val_labels = (val_preds >= 0.5).float()
        val_acc = (val_labels == y_val_t).float().mean().item()

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(
            f"Epoch {epoch+1:3d}/{num_epochs} | "
            f"train_loss={train_loss:.6f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.6f} | val_acc={val_acc:.4f}"
        )


# ----------------------------
# 8. Example inference
# ----------------------------
model.eval()
with torch.no_grad():
    sample_probs = model(X_val_t[:10])
    sample_pred = (sample_probs >= 0.5).float()

print("\nFirst 10 validation predictions:")
for i in range(10):
    print(
        f"gt={y_val_t[i].item():.0f}, "
        f"prob={sample_probs[i].item():.4f}, "
        f"pred={sample_pred[i].item():.0f}"
    )

# ----------------------------
# 10. Final evaluation metrics
# ----------------------------
model.eval()

with torch.no_grad():
    probs = model(X_val_t)                  # (N, 1)
    preds = (probs >= 0.5).float()

    y_true = y_val_t
    y_pred = preds

    # Accuracy
    accuracy = (y_pred == y_true).float().mean().item()

    # Confusion matrix components
    TP = ((y_pred == 1) & (y_true == 1)).sum().item() / (y_true == 1).sum().item()
    TN = ((y_pred == 0) & (y_true == 0)).sum().item() /(y_true == 0).sum().item()
    FP = ((y_pred == 1) & (y_true == 0)).sum().item() / (y_true == 0).sum().item()
    FN = ((y_pred == 0) & (y_true == 1)).sum().item() / (y_true == 1).sum().item()

    # Precision / Recall / F1 (safe divide)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

print("\n===== Final Evaluation =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")


print("\n===== Layer-wise Weights =====")

for name, param in model.named_parameters():
    print(f"\n{name}")
    print(f"shape: {tuple(param.shape)}")
    print(param.detach().cpu().numpy())

# ----------------------------
# 9. Save model and normalization
# ----------------------------
save_path = Path(__file__).resolve().parent / "bch_nn_model.pt"
torch.save({
    "model_state_dict": model.state_dict(),
    "input_dim": input_dim,
    "hidden_dim": hidden_dim,
}, save_path)

print(f"\nSaved model to: {save_path}")


export_path = Path(__file__).resolve().parent / "bch_nn_weights_small_p5.txt"

fc1_w = model.fc1.weight.detach().cpu().numpy()   # shape: (hidden_dim, input_dim)
fc1_b = model.fc1.bias.detach().cpu().numpy()     # shape: (hidden_dim,)
fc2_w = model.fc2.weight.detach().cpu().numpy()   # shape: (1, hidden_dim)
fc2_b = model.fc2.bias.detach().cpu().numpy()     # shape: (1,)


with open(export_path, "w") as f:
    input_dim = fc1_w.shape[1]
    hidden_dim = fc1_w.shape[0]

    f.write(f"{input_dim} {hidden_dim}\n")

    # fc1
    f.write("fc1_weight\n")
    for row in fc1_w:
        f.write(" ".join(map(str, row.tolist())) + "\n")

    f.write("fc1_bias\n")
    f.write(" ".join(map(str, fc1_b.tolist())) + "\n")

    # fc2
    f.write("fc2_weight\n")
    f.write(" ".join(map(str, fc2_w[0].tolist())) + "\n")

    f.write("fc2_bias\n")
    f.write(str(fc2_b[0]) + "\n")

print(f"Exported weights to: {export_path}")