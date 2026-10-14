
from pathlib import Path
import random
import copy
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, average_precision_score


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "temporal_data"
CACHE_DIR = DATA_DIR / "full_feature_cache"
OUTPUT_DIR = DATA_DIR / "gru_small_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 2026
NUM_CLASSES = 19
FEATURE_DIM = 2048
HIDDEN_DIM = 32

BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 12
MIN_DELTA = 1e-4
LEARNING_RATE = 1e-3

DEVICE = torch.device("cpu")

EXPECTED_COUNTS = {
    "train": 14812,
    "validation": 7703,
    "test": 7496,
}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 70)
print("GRU TEMPORAL CLASSIFIER")
print("=" * 70)
print("Device:", DEVICE)


# ============================================================
# 2. LOAD CACHED FEATURES
# ============================================================

def load_split(split_name):
    path = CACHE_DIR / (
        f"temporal_features_{split_name}_seed{SEED}.npz"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing temporal cache: {path}\n"
            "Run scripts/29_extract_temporal_features.py first."
        )

    with np.load(path, allow_pickle=False) as data:
        X = data["X_temporal"].astype(np.float32)
        Y = data["Y"].astype(np.float32)

    if X.shape != (
        EXPECTED_COUNTS[split_name], 4, FEATURE_DIM
    ):
        raise ValueError(
            f"Unexpected {split_name} feature shape: {X.shape}"
        )

    if Y.shape != (
        EXPECTED_COUNTS[split_name], NUM_CLASSES
    ):
        raise ValueError(
            f"Unexpected {split_name} label shape: {Y.shape}"
        )

    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError(f"Non-finite values in {split_name} data.")

    if not np.isin(Y, [0, 1]).all():
        raise ValueError(f"Non-binary labels in {split_name} data.")

    print(
        f"{split_name.capitalize()}: "
        f"features {X.shape}, labels {Y.shape}"
    )

    return X, Y


print("\nLoading feature caches...")

X_train, Y_train = load_split("train")
X_val, Y_val = load_split("validation")
X_test, Y_test = load_split("test")


# ============================================================
# 3. NORMALIZE USING TRAINING DATA ONLY
# ============================================================

print("\nNormalizing features using training data only...")

# Each feature dimension gets one mean and standard deviation,
# calculated across training samples and dates.
feature_mean = X_train.mean(axis=(0, 1), keepdims=True)
feature_std = X_train.std(axis=(0, 1), keepdims=True)

feature_std[feature_std < 1e-6] = 1.0

X_train = (X_train - feature_mean) / feature_std
X_val = (X_val - feature_mean) / feature_std
X_test = (X_test - feature_mean) / feature_std

print("Normalization complete.")


# ============================================================
# 4. CALCULATE CLASS WEIGHTS FROM TRAINING LABELS
# ============================================================

positive_counts = Y_train.sum(axis=0)
negative_counts = len(Y_train) - positive_counts

# Avoid division by zero for any class.
pos_weight = negative_counts / np.maximum(positive_counts, 1)

# Limit extreme weights to keep training stable.
pos_weight = np.clip(pos_weight, 1.0, 20.0).astype(np.float32)

print("\nPositive counts by class:")
print(positive_counts.astype(int))

print("\nPositive loss weights:")
print(np.round(pos_weight, 3))


# ============================================================
# 5. BUILD DATA LOADERS
# ============================================================

def make_loader(X, Y, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(Y.astype(np.float32))
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0
    )


train_loader = make_loader(X_train, Y_train, True)
val_loader = make_loader(X_val, Y_val, False)


# ============================================================
# 6. DEFINE THE GRU MODEL
# ============================================================

class GRUTemporalClassifier(nn.Module):

    def __init__(
        self,
        feature_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Input: (batch, 4 dates, 2048 features)
        _, hidden = self.gru(x)

        # Final hidden state summarizes the ordered sequence.
        final_hidden = hidden[-1]

        return self.classifier(
            self.dropout(final_hidden)
        )


model = GRUTemporalClassifier().to(DEVICE)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print(f"\nTrainable GRU parameters: {parameter_count:,}")


# ============================================================
# 7. LOSS AND OPTIMIZER
# ============================================================

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.from_numpy(pos_weight).to(DEVICE)
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 8. TRAINING AND VALIDATION
# ============================================================

history = []
best_val_loss = float("inf")
best_epoch = 0
best_state = None
epochs_without_improvement = 0

print("\nStarting training...")

for epoch in range(1, MAX_EPOCHS + 1):

    model.train()
    train_loss_sum = 0.0
    train_samples = 0

    for features, targets in train_loader:
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        logits = model(features)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        batch_count = len(features)
        train_loss_sum += loss.item() * batch_count
        train_samples += batch_count

    train_loss = train_loss_sum / train_samples

    model.eval()
    val_loss_sum = 0.0
    val_samples = 0

    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)

            logits = model(features)
            loss = criterion(logits, targets)

            batch_count = len(features)
            val_loss_sum += loss.item() * batch_count
            val_samples += batch_count

    val_loss = val_loss_sum / val_samples

    improved = val_loss < best_val_loss - MIN_DELTA

    if improved:
        best_val_loss = val_loss
        best_epoch = epoch

        best_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }

        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": val_loss,
    })

    if epoch == 1 or epoch % 5 == 0 or improved:
        marker = " *BEST*" if improved else ""

        print(
            f"Epoch {epoch:03d} | "
            f"Train loss: {train_loss:.5f} | "
            f"Validation loss: {val_loss:.5f}"
            f"{marker}"
        )

    if epochs_without_improvement >= PATIENCE:
        print(
            f"Early stopping at epoch {epoch}; "
            f"best epoch was {best_epoch}."
        )
        break


if best_state is None:
    raise RuntimeError("No valid GRU checkpoint was produced.")

model.load_state_dict(best_state)

checkpoint_path = OUTPUT_DIR / "gru_temporal_seed2026.pt"

torch.save({
    "state_dict": best_state,
    "seed": SEED,
    "best_epoch": best_epoch,
    "best_validation_loss": best_val_loss,
    "feature_dim": FEATURE_DIM,
    "hidden_dim": HIDDEN_DIM,
    "num_classes": NUM_CLASSES,
    "feature_mean": feature_mean,
    "feature_std": feature_std,
    "pos_weight": pos_weight,
}, checkpoint_path)

pd.DataFrame(history).to_csv(
    OUTPUT_DIR / "training_history.csv",
    index=False
)

print("\nBest checkpoint:", checkpoint_path)
print("Best epoch:", best_epoch)
print("Best validation loss:", round(best_val_loss, 6))


# ============================================================
# 9. FINAL TEST EVALUATION
# ============================================================

# The test split is not used to select epochs or tune parameters.
print("\nEvaluating selected checkpoint on the test split...")

test_loader = make_loader(X_test, Y_test, False)

model.eval()
test_loss_sum = 0.0
test_samples = 0
probability_batches = []

with torch.no_grad():
    for features, targets in test_loader:
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        logits = model(features)
        loss = criterion(logits, targets)

        batch_count = len(features)
        test_loss_sum += loss.item() * batch_count
        test_samples += batch_count

        probability_batches.append(
            torch.sigmoid(logits).cpu().numpy()
        )

test_loss = test_loss_sum / test_samples
probabilities = np.concatenate(probability_batches, axis=0)

y_true = Y_test.astype(int)
y_pred = (probabilities >= 0.5).astype(int)

macro_f1 = f1_score(
    y_true, y_pred, average="macro", zero_division=0
)

micro_f1 = f1_score(
    y_true, y_pred, average="micro", zero_division=0
)

per_class_ap = []

for class_id in range(NUM_CLASSES):
    truth = y_true[:, class_id]

    if truth.sum() > 0:
        per_class_ap.append(
            average_precision_score(
                truth, probabilities[:, class_id]
            )
        )

mean_ap = float(np.mean(per_class_ap))

results = {
    "experiment": "GRU temporal classifier",
    "seed": SEED,
    "best_epoch": best_epoch,
    "best_validation_loss": best_val_loss,
    "test_loss": test_loss,
    "macro_f1": float(macro_f1),
    "micro_f1": float(micro_f1),
    "mAP": mean_ap,
    "classes_in_mAP": len(per_class_ap),
    "threshold": 0.5,
}

with open(
    OUTPUT_DIR / "test_metrics.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(results, file, indent=4)

print("\n" + "=" * 70)
print("GRU TEST RESULTS")
print("=" * 70)
print(f"Test loss:  {test_loss:.6f}")
print(f"Macro-F1:   {macro_f1:.6f}")
print(f"Micro-F1:   {micro_f1:.6f}")
print(f"mAP:        {mean_ap:.6f}")
print(f"Best epoch: {best_epoch}")
print("Saved results:", OUTPUT_DIR / "test_metrics.json")
print("\nGRU EXPERIMENT COMPLETE")
