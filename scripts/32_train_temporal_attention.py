
from pathlib import Path
import copy
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score,
    average_precision_score,
)


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

SEED = 2026
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 12
MIN_DELTA = 1e-4
LEARNING_RATE = 1e-3
THRESHOLD = 0.5

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "temporal_data"
CACHE_DIR = DATA_DIR / "full_feature_cache"

OUTPUT_DIR = DATA_DIR / "attention_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = CACHE_DIR / "temporal_features_train_seed2026.npz"
VAL_FILE = CACHE_DIR / "temporal_features_validation_seed2026.npz"
TEST_FILE = CACHE_DIR / "temporal_features_test_seed2026.npz"


# ------------------------------------------------------------
# REPRODUCIBILITY
# ------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

class TemporalAttentionClassifier(nn.Module):
    def __init__(self, feature_dim=2048, num_classes=19):
        super().__init__()

        # Produces one attention score for each date.
        self.attention = nn.Linear(feature_dim, 1)

        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        # x: (batch, 4 dates, 2048 features)

        scores = self.attention(x).squeeze(-1)
        # scores: (batch, 4)

        weights = torch.softmax(scores, dim=1)
        # weights: (batch, 4)

        pooled = torch.sum(
            x * weights.unsqueeze(-1),
            dim=1,
        )
        # pooled: (batch, 2048)

        logits = self.classifier(self.dropout(pooled))

        return logits


# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------

def load_cache(path):
    with np.load(path) as data:
        X = data["X_temporal"].astype(np.float32)
        Y = data["Y"].astype(np.float32)

    if X.ndim != 3 or X.shape[1:] != (4, 2048):
        raise ValueError(
            f"Unexpected feature shape in {path}: {X.shape}"
        )

    if Y.shape != (len(X), 19):
        raise ValueError(
            f"Unexpected label shape in {path}: {Y.shape}"
        )

    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError(f"Non-finite values found in {path}")

    return X, Y


def make_loader(X, Y, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(Y),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )


# ------------------------------------------------------------
# LOSS AND EVALUATION
# ------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0

    for features, labels in loader:
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = len(features)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def calculate_metrics(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(features)
            loss = criterion(logits, labels)

            probabilities = torch.sigmoid(logits)

            batch_size = len(features)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_probabilities.append(
                probabilities.cpu().numpy()
            )
            all_labels.append(labels.cpu().numpy())

    probabilities = np.concatenate(all_probabilities)
    labels = np.concatenate(all_labels)

    predictions = (
        probabilities >= THRESHOLD
    ).astype(np.int32)

    macro_f1 = f1_score(
        labels, predictions,
        average="macro",
        zero_division=0,
    )

    micro_f1 = f1_score(
        labels, predictions,
        average="micro",
        zero_division=0,
    )

    # Compute AP for classes with both positive and
    # negative examples in this evaluation split.
    valid_classes = [
        c for c in range(labels.shape[1])
        if np.unique(labels[:, c]).size == 2
    ]

    mean_ap = average_precision_score(
        labels[:, valid_classes],
        probabilities[:, valid_classes],
        average="macro",
    )

    return {
        "test_loss": total_loss / total_samples,
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "mAP": float(mean_ap),
        "classes_in_mAP": len(valid_classes),
        "threshold": THRESHOLD,
    }


# ------------------------------------------------------------
# MAIN EXPERIMENT
# ------------------------------------------------------------

def main():
    print("=" * 70)
    print("TEMPORAL ATTENTION CLASSIFIER")
    print("=" * 70)
    print("Device:", DEVICE)

    print("\nLoading temporal feature caches...")

    X_train, Y_train = load_cache(TRAIN_FILE)
    X_val, Y_val = load_cache(VAL_FILE)
    X_test, Y_test = load_cache(TEST_FILE)

    print("Train:", X_train.shape, Y_train.shape)
    print("Validation:", X_val.shape, Y_val.shape)
    print("Test:", X_test.shape, Y_test.shape)

    # --------------------------------------------------------
    # NORMALIZE USING TRAINING DATA ONLY
    # --------------------------------------------------------

    print("\nNormalizing features using training data only...")

    feature_mean = X_train.mean(
        axis=(0, 1), keepdims=True
    )

    feature_std = X_train.std(
        axis=(0, 1), keepdims=True
    )

    feature_std = np.maximum(feature_std, 1e-6)

    X_train = (X_train - feature_mean) / feature_std
    X_val = (X_val - feature_mean) / feature_std
    X_test = (X_test - feature_mean) / feature_std

    print("Normalization complete.")

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    positive_counts = Y_train.sum(axis=0)
    negative_counts = len(Y_train) - positive_counts

    pos_weight = negative_counts / np.maximum(
        positive_counts, 1
    )

    pos_weight = np.clip(pos_weight, 1.0, 20.0)

    print("\nPositive loss weights:")
    print(np.round(pos_weight, 3))

    pos_weight_tensor = torch.tensor(
        pos_weight,
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight_tensor
    )

    train_loader = make_loader(
        X_train, Y_train, shuffle=True
    )
    val_loader = make_loader(
        X_val, Y_val, shuffle=False
    )
    test_loader = make_loader(
        X_test, Y_test, shuffle=False
    )

    # --------------------------------------------------------
    # MODEL AND OPTIMIZER
    # --------------------------------------------------------

    model = TemporalAttentionClassifier().to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nTrainable attention-model parameters:", parameter_count)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []

    checkpoint_path = (
        OUTPUT_DIR / "temporal_attention_seed2026.pt"
    )

    history_path = OUTPUT_DIR / "training_history.csv"
    metrics_path = OUTPUT_DIR / "test_metrics.json"

    # --------------------------------------------------------
    # TRAINING WITH VALIDATION-BASED EARLY STOPPING
    # --------------------------------------------------------

    print("\nStarting training...")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_loss = run_epoch(
            model,
            val_loader,
            criterion,
        )

        improved = val_loss < best_val_loss - MIN_DELTA

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": best_state,
                    "seed": SEED,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_val_loss,
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "pos_weight": pos_weight,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
        })

        if epoch <= 5 or epoch % 5 == 0 or improved:
            marker = " *BEST*" if improved else ""
            print(
                f"Epoch {epoch:03d} | "
                f"Train loss: {train_loss:.5f} | "
                f"Validation loss: {val_loss:.5f}"
                f"{marker}"
            )

        if epochs_without_improvement >= PATIENCE:
            print(
                f"\nEarly stopping at epoch {epoch}; "
                f"best epoch was {best_epoch}."
            )
            break

    if best_state is None:
        raise RuntimeError("No valid checkpoint was produced.")

    pd.DataFrame(history).to_csv(
        history_path, index=False
    )

    print("\nBest checkpoint:", checkpoint_path)
    print("Best epoch:", best_epoch)
    print(f"Best validation loss: {best_val_loss:.5f}")

    # --------------------------------------------------------
    # FINAL TEST EVALUATION
    # --------------------------------------------------------

    print("\nEvaluating selected checkpoint on test split...")

    model.load_state_dict(best_state)

    results = calculate_metrics(
        model,
        test_loader,
        criterion,
    )

    results.update({
        "experiment": "Learned temporal attention",
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "trainable_parameters": parameter_count,
    })

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("\n" + "=" * 70)
    print("TEMPORAL ATTENTION TEST RESULTS")
    print("=" * 70)
    print(f"Test loss:  {results['test_loss']:.6f}")
    print(f"Macro-F1:   {results['macro_f1']:.6f}")
    print(f"Micro-F1:   {results['micro_f1']:.6f}")
    print(f"mAP:        {results['mAP']:.6f}")
    print("Best epoch:", best_epoch)
    print("Saved results:", metrics_path)
    print("\nTEMPORAL ATTENTION EXPERIMENT COMPLETE")


if __name__ == "__main__":
    main()
