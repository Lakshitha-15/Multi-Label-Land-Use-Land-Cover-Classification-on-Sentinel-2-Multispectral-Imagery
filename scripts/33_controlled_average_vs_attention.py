from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    f1_score,
    average_precision_score,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "temporal_data"
CACHE_DIR = DATA_DIR / "full_feature_cache"
OUTPUT_DIR = DATA_DIR / "controlled_comparison"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 2026
FEATURE_DIM = 2048
NUM_CLASSES = 19
MAX_EPOCHS = 250
PATIENCE = 12
MIN_DELTA = 1e-4
LEARNING_RATE = 1e-3
THRESHOLD = 0.5

DEVICE = torch.device("cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# 2. LOAD TEMPORAL FEATURE CACHES
# ============================================================

def load_split(split):
    path = CACHE_DIR / (
        f"temporal_features_{split}_seed2026.npz"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        X = data["X_temporal"].astype(np.float32)
        Y = data["Y"].astype(np.float32)

    if X.ndim != 3 or X.shape[1:] != (4, FEATURE_DIM):
        raise ValueError(
            f"Unexpected {split} feature shape: {X.shape}"
        )

    if Y.shape != (len(X), NUM_CLASSES):
        raise ValueError(
            f"Unexpected {split} label shape: {Y.shape}"
        )

    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError(f"Non-finite values in {split} data.")

    print(f"{split}: X={X.shape}, Y={Y.shape}")
    return X, Y


X_train_seq, y_train = load_split("train")
X_val_seq, y_val = load_split("validation")
X_test_seq, y_test = load_split("test")


# ============================================================
# 3. DERIVE AVERAGING FEATURES FROM THE SAME SEQUENCES
# ============================================================

X_train_avg = X_train_seq.mean(axis=1)
X_val_avg = X_val_seq.mean(axis=1)
X_test_avg = X_test_seq.mean(axis=1)


# ============================================================
# 4. NORMALIZE BOTH METHODS USING TRAINING DATA ONLY
# ============================================================

# Use the same training observations and dates to estimate
# one mean and standard deviation for each feature dimension.
feature_mean = X_train_seq.mean(axis=(0, 1), keepdims=True)
feature_std = X_train_seq.std(axis=(0, 1), keepdims=True)
feature_std = np.maximum(feature_std, 1e-6)

def normalize_sequences(X):
    return (X - feature_mean) / feature_std

X_train_seq = normalize_sequences(X_train_seq)
X_val_seq = normalize_sequences(X_val_seq)
X_test_seq = normalize_sequences(X_test_seq)

# Average after normalizing so both methods share exactly
# the same normalized temporal inputs.
X_train_avg = X_train_seq.mean(axis=1)
X_val_avg = X_val_seq.mean(axis=1)
X_test_avg = X_test_seq.mean(axis=1)


# ============================================================
# 5. CLASS WEIGHTS
# ============================================================

positive_counts = y_train.sum(axis=0)
negative_counts = len(y_train) - positive_counts

pos_weight = np.clip(
    negative_counts / np.maximum(positive_counts, 1),
    1.0,
    20.0,
)

pos_weight_tensor = torch.tensor(
    pos_weight, dtype=torch.float32, device=DEVICE
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight_tensor
)


# ============================================================
# 6. MODELS
# ============================================================

class AverageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(FEATURE_DIM, NUM_CLASSES)

    def forward(self, X):
        return self.classifier(X)


class TemporalAttentionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Linear(FEATURE_DIM, 1)
        self.classifier = nn.Linear(FEATURE_DIM, NUM_CLASSES)

    def forward(self, X):
        scores = self.attention(X).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = (X * weights.unsqueeze(-1)).sum(dim=1)
        return self.classifier(pooled)


# ============================================================
# 7. TRAINING AND VALIDATION
# ============================================================

def train_model(name, model, Xtr, Xva):
    print(f"\nTraining {name}...")

    torch.manual_seed(SEED)
    model = model.to(DEVICE)

    Xtr_t = torch.from_numpy(Xtr).to(DEVICE)
    Xva_t = torch.from_numpy(Xva).to(DEVICE)
    ytr_t = torch.from_numpy(y_train).to(DEVICE)
    yva_t = torch.from_numpy(y_val).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE
    )

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    no_improvement = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(Xtr_t)
        train_loss = criterion(logits, ytr_t)

        if not torch.isfinite(train_loss):
            raise RuntimeError("Non-finite training loss.")

        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(
                model(Xva_t), yva_t
            ).item()

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss.item()),
            "validation_loss": float(val_loss),
        })

        if val_loss < best_loss - MIN_DELTA:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            no_improvement = 0
        else:
            no_improvement += 1

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train={train_loss.item():.4f} | "
                f"validation={val_loss:.4f}"
            )

        if no_improvement >= PATIENCE:
            print(f"Early stopping; best epoch={best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("No best checkpoint was saved.")

    model.load_state_dict(best_state)

    torch.save(
        {
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "seed": SEED,
        },
        OUTPUT_DIR / f"{name}_checkpoint.pt",
    )

    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / f"{name}_history.csv", index=False
    )

    return model, best_epoch, best_loss


# ============================================================
# 8. TRAIN BOTH METHODS
# ============================================================

average_model, avg_epoch, avg_val_loss = train_model(
    "uniform_average",
    AverageClassifier(),
    X_train_avg,
    X_val_avg,
)

attention_model, att_epoch, att_val_loss = train_model(
    "temporal_attention",
    TemporalAttentionClassifier(),
    X_train_seq,
    X_val_seq,
)


# ============================================================
# 9. TEST EVALUATION — AFTER CHECKPOINT SELECTION
# ============================================================

def evaluate(model, X_test):
    model.eval()

    X_t = torch.from_numpy(X_test).to(DEVICE)
    y_t = torch.from_numpy(y_test).to(DEVICE)

    with torch.no_grad():
        logits = model(X_t)
        test_loss = criterion(logits, y_t).item()
        probabilities = torch.sigmoid(logits).cpu().numpy()

    predictions = (probabilities >= THRESHOLD).astype(np.int32)
    truth = y_test.astype(np.int32)

    valid_classes = truth.sum(axis=0) > 0

    return {
        "test_loss": float(test_loss),
        "macro_f1": float(f1_score(
            truth, predictions,
            average="macro", zero_division=0
        )),
        "micro_f1": float(f1_score(
            truth, predictions,
            average="micro", zero_division=0
        )),
        "mAP": float(average_precision_score(
            truth[:, valid_classes],
            probabilities[:, valid_classes],
            average="macro",
        )),
        "threshold": THRESHOLD,
    }


results = {
    "uniform_average": {
        "best_epoch": avg_epoch,
        "best_validation_loss": avg_val_loss,
        **evaluate(average_model, X_test_avg),
    },
    "temporal_attention": {
        "best_epoch": att_epoch,
        "best_validation_loss": att_val_loss,
        **evaluate(attention_model, X_test_seq),
    },
}

with open(OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 65)
print("CONTROLLED COMPARISON RESULTS")
print("=" * 65)

for name, metrics in results.items():
    print(f"\n{name}")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

print(f"\nResults saved to: {OUTPUT_DIR}")