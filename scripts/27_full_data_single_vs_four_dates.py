
from pathlib import Path
import importlib.util
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
)


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"
DATA_DIR = PROJECT_DIR / "temporal_data"

CACHE_DIR = DATA_DIR / "full_feature_cache"
CHECKPOINT_DIR = DATA_DIR / "full_data_checkpoints"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 1
MAX_EPOCHS = 250
PATIENCE = 12
MIN_DELTA = 1e-4
LEARNING_RATE = 1e-3

SEED = 2026
NUM_CLASSES = 19
FEATURE_DIM = 2048

EXPECTED_COUNTS = {
    "train": 14812,
    "validation": 7703,
    "test": 7496,
}

device = torch.device("cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 72)
print("FULL-DATA SINGLE-DATE VS FOUR-DATE EXPERIMENT")
print("=" * 72)
print("Device:", device)
print("Maximum epochs:", MAX_EPOCHS)


# ============================================================
# 2. LOAD EXISTING PROJECT MODULES
# ============================================================

def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(
        module_name, file_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


dataset_module = load_module(
    "full_temporal_dataset_module",
    SCRIPTS_DIR / "17_build_pytorch_dataset.py"
)

model_module = load_module(
    "full_temporal_model_module",
    SCRIPTS_DIR / "23_test_pretrained_temporal_model.py"
)


# ============================================================
# 3. LOAD ALL THREE OFFICIAL SPLITS
# ============================================================

print("\nPreparing datasets...")

train_dataset = dataset_module.TemporalLULCDataset(
    split="train"
)

val_dataset = dataset_module.TemporalLULCDataset(
    split="validation"
)

test_dataset = dataset_module.TemporalLULCDataset(
    split="test"
)

datasets = {
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset,
}

for split_name, dataset in datasets.items():
    actual_count = len(dataset)
    expected_count = EXPECTED_COUNTS[split_name]

    print(f"{split_name.capitalize()} locations: {actual_count:,}")

    if actual_count != expected_count:
        raise ValueError(
            f"{split_name} has {actual_count} locations; "
            f"expected {expected_count}. Check the manifest "
            "and dataset implementation before proceeding."
        )

print("\nAll split counts verified.")


# ============================================================
# 4. FEATURE CACHE PATHS
# ============================================================

def get_cache_path(split_name):
    return CACHE_DIR / (
        f"full_features_{split_name}_seed{SEED}.npz"
    )


# ============================================================
# 5. BUILD FROZEN PRETRAINED FEATURE EXTRACTOR
# ============================================================

def build_feature_extractor():
    print("\nLoading pretrained ResNet50...")

    model = model_module.TemporalResNet50(
        num_classes=NUM_CLASSES,
        pretrained=True
    ).to(device)

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model


# ============================================================
# 6. EXTRACT FEATURES FOR ONE COMPLETE SPLIT
# ============================================================

def extract_features(model, dataset, split_name):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    single_features = []
    four_date_features = []
    targets_list = []

    total = len(loader)

    print(f"\nExtracting {split_name} features...")
    print(f"Sequences to process: {total:,}")

    with torch.no_grad():
        for batch_number, (images, targets) in enumerate(
            loader, start=1
        ):
            images = images.to(device)
            targets = targets.float().to(device)

            if tuple(images.shape[1:]) != (4, 6, 120, 120):
                raise ValueError(
                    f"Unexpected image shape: {tuple(images.shape)}"
                )

            batch_size, num_dates, num_bands, height, width = (
                images.shape
            )

            # Process all four dates through the same backbone.
            flattened = images.reshape(
                batch_size * num_dates,
                num_bands,
                height,
                width
            )

            features = model.feature_extractor(flattened)

            features = features.reshape(
                batch_size, num_dates, -1
            )

            # Baseline: first observation only.
            single_features.append(
                features[:, 0, :].cpu().numpy()
            )

            # Proposed method: average four observation features.
            four_date_features.append(
                features.mean(dim=1).cpu().numpy()
            )

            targets_list.append(targets.cpu().numpy())

            if batch_number % 256 == 0 or batch_number == total:
                print(
                    f"  Processed {batch_number:,}/{total:,}"
                )

    X_single = np.concatenate(single_features, axis=0)
    X_four = np.concatenate(four_date_features, axis=0)
    Y = np.concatenate(targets_list, axis=0)

    if X_single.shape != (len(dataset), FEATURE_DIM):
        raise ValueError(
            f"Unexpected single-date feature shape: "
            f"{X_single.shape}"
        )

    if X_four.shape != (len(dataset), FEATURE_DIM):
        raise ValueError(
            f"Unexpected four-date feature shape: {X_four.shape}"
        )

    if Y.shape != (len(dataset), NUM_CLASSES):
        raise ValueError(f"Unexpected target shape: {Y.shape}")

    return X_single, X_four, Y


# ============================================================
# 7. LOAD OR EXTRACT EACH SPLIT'S FEATURES
# ============================================================

def load_or_extract_split(model, split_name, dataset):
    cache_path = get_cache_path(split_name)

    expected_shapes = (
        (len(dataset), FEATURE_DIM),
        (len(dataset), FEATURE_DIM),
        (len(dataset), NUM_CLASSES),
    )

    if cache_path.exists():
        print(f"\nFound cache for {split_name}:")
        print(cache_path)

        with np.load(cache_path, allow_pickle=False) as cache:
            X_single = cache["X_single"]
            X_four = cache["X_four"]
            Y = cache["Y"]

        actual_shapes = (
            X_single.shape,
            X_four.shape,
            Y.shape,
        )

        if actual_shapes != expected_shapes:
            raise ValueError(
                f"Cache shape mismatch for {split_name}. "
                f"Expected {expected_shapes}, got {actual_shapes}. "
                "Move the invalid cache file elsewhere and rerun."
            )

        print("Cached features loaded.")

    else:
        X_single, X_four, Y = extract_features(
            model, dataset, split_name
        )

        # Save this split immediately so completed extraction
        # can be reused if a later stage is interrupted.
        np.savez(
            cache_path,
            X_single=X_single,
            X_four=X_four,
            Y=Y,
        )

        print(f"\nSaved {split_name} feature cache:")
        print(cache_path)

    for name, array in (
        ("single-date features", X_single),
        ("four-date features", X_four),
        ("labels", Y),
    ):
        if not np.isfinite(array).all():
            raise ValueError(
                f"Non-finite values found in {split_name} {name}."
            )

    if not np.isin(Y, [0.0, 1.0]).all():
        raise ValueError(
            f"Labels in {split_name} are not binary."
        )

    return X_single, X_four, Y


print("\nChecking feature caches...")

missing_cache = any(
    not get_cache_path(split_name).exists()
    for split_name in datasets
)

if missing_cache:
    feature_model = build_feature_extractor()
else:
    feature_model = None
    print("All three feature caches already exist.")

features = {}

for split_name, dataset in datasets.items():
    features[split_name] = load_or_extract_split(
        feature_model,
        split_name,
        dataset
    )

# The frozen backbone is no longer needed after extraction.
del feature_model

X_train_single, X_train_four, y_train = features["train"]
X_val_single, X_val_four, y_val = features["validation"]
X_test_single, X_test_four, y_test = features["test"]

del features

print("\nFeature extraction/loading completed.")
print("Training:", X_train_single.shape, y_train.shape)
print("Validation:", X_val_single.shape, y_val.shape)
print("Test:", X_test_single.shape, y_test.shape)


# ============================================================
# 8. LOAD CLASS WEIGHTS FROM THE FULL TRAINING SPLIT
# ============================================================

class_df = pd.read_csv(
    DATA_DIR / "class_distribution.csv"
).sort_values("class_id").reset_index(drop=True)

if class_df["class_id"].tolist() != list(range(NUM_CLASSES)):
    raise ValueError("Class IDs must be 0 through 18.")

train_counts = torch.tensor(
    class_df["train_count"].to_numpy(),
    dtype=torch.float32
)

if (train_counts <= 0).any():
    raise ValueError(
        "At least one class has zero training positives."
    )

if (train_counts > len(train_dataset)).any():
    raise ValueError(
        "A class positive count exceeds the training-set size."
    )

pos_weight = (
    (len(train_dataset) - train_counts) / train_counts
).clamp(min=1.0, max=20.0)

print("\nFull-training class weights loaded.")


# ============================================================
# 9. TRAIN A CLASSIFIER AND RESTORE ITS BEST CHECKPOINT
# ============================================================

def train_classifier(
    X_train,
    Y_train,
    X_val,
    Y_val,
    experiment_name,
    checkpoint_path
):
    print("\n" + "=" * 72)
    print("Training:", experiment_name)
    print("=" * 72)

    # Both classifiers start from the same random initialization.
    torch.manual_seed(SEED)

    X_train_t = torch.from_numpy(
        X_train.astype(np.float32, copy=False)
    ).to(device)

    Y_train_t = torch.from_numpy(
        Y_train.astype(np.float32, copy=False)
    ).to(device)

    X_val_t = torch.from_numpy(
        X_val.astype(np.float32, copy=False)
    ).to(device)

    Y_val_t = torch.from_numpy(
        Y_val.astype(np.float32, copy=False)
    ).to(device)

    classifier = nn.Linear(
        FEATURE_DIM, NUM_CLASSES
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(device)
    )

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=LEARNING_RATE
    )

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    epochs_trained = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        classifier.train()
        optimizer.zero_grad()

        train_logits = classifier(X_train_t)
        train_loss = criterion(train_logits, Y_train_t)

        if not torch.isfinite(train_loss):
            raise RuntimeError("Non-finite training loss.")

        train_loss.backward()
        optimizer.step()

        classifier.eval()

        with torch.no_grad():
            val_logits = classifier(X_val_t)
            val_loss = criterion(
                val_logits, Y_val_t
            ).item()

        epochs_trained = epoch

        improved = val_loss < best_loss - MIN_DELTA

        if improved:
            best_loss = val_loss
            best_epoch = epoch

            best_state = {
                name: value.detach().cpu().clone()
                for name, value in classifier.state_dict().items()
            }

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            epoch == 1
            or epoch % 5 == 0
            or improved
            or epochs_without_improvement == PATIENCE
        ):
            marker = " *best*" if improved else ""

            print(
                f"Epoch {epoch:03d} | "
                f"Train loss: {train_loss.item():.4f} | "
                f"Validation loss: {val_loss:.4f}"
                f"{marker}"
            )

        if epochs_without_improvement >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch was {best_epoch}."
            )
            break

    if best_state is None:
        raise RuntimeError("No valid best checkpoint was saved.")

    classifier.load_state_dict(best_state)
    classifier.eval()

    torch.save(
        {
            "state_dict": best_state,
            "experiment": experiment_name,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "seed": SEED,
            "feature_dim": FEATURE_DIM,
            "num_classes": NUM_CLASSES,
        },
        checkpoint_path
    )

    print("\nBest checkpoint saved:")
    print(checkpoint_path)
    print("Best epoch:", best_epoch)
    print("Best validation loss:", round(best_loss, 6))

    # Validation metrics are reported separately from test metrics.
    with torch.no_grad():
        val_probabilities = torch.sigmoid(
            classifier(X_val_t)
        ).cpu().numpy()

    y_val_int = Y_val.astype(int)
    val_predictions = (val_probabilities >= 0.5).astype(int)

    val_macro_f1 = f1_score(
        y_val_int,
        val_predictions,
        average="macro",
        zero_division=0
    )

    val_micro_f1 = f1_score(
        y_val_int,
        val_predictions,
        average="micro",
        zero_division=0
    )

    val_aps = []

    for class_id in range(NUM_CLASSES):
        if y_val_int[:, class_id].sum() > 0:
            val_aps.append(
                average_precision_score(
                    y_val_int[:, class_id],
                    val_probabilities[:, class_id]
                )
            )

    val_map = float(np.mean(val_aps)) if val_aps else float("nan")

    validation_result = {
        "experiment": experiment_name,
        "best_epoch": best_epoch,
        "epochs_trained": epochs_trained,
        "best_validation_loss": best_loss,
        "validation_macro_f1": val_macro_f1,
        "validation_micro_f1": val_micro_f1,
        "validation_mAP": val_map,
        "validation_classes_in_mAP": len(val_aps),
    }

    print(f"Validation Macro-F1: {val_macro_f1:.4f}")
    print(f"Validation Micro-F1: {val_micro_f1:.4f}")
    print(f"Validation mAP: {val_map:.4f}")

    return classifier, validation_result


# ============================================================
# 10. EVALUATE ON THE HELD-OUT TEST SPLIT
# ============================================================

def evaluate_test(
    classifier,
    X_test,
    Y_test,
    experiment_name
):
    print("\nEvaluating test split:", experiment_name)

    X_test_t = torch.from_numpy(
        X_test.astype(np.float32, copy=False)
    ).to(device)

    Y_test_t = torch.from_numpy(
        Y_test.astype(np.float32, copy=False)
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(device)
    )

    classifier.eval()

    with torch.no_grad():
        logits = classifier(X_test_t)
        test_loss = criterion(logits, Y_test_t).item()
        probabilities = torch.sigmoid(logits).cpu().numpy()

    y_true = Y_test.astype(int)
    y_pred = (probabilities >= 0.5).astype(int)

    macro_f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    micro_f1 = f1_score(
        y_true, y_pred, average="micro", zero_division=0
    )

    aps = []

    per_class_rows = []

    for class_id in range(NUM_CLASSES):
        truth = y_true[:, class_id]
        prediction = y_pred[:, class_id]
        scores = probabilities[:, class_id]

        support = int(truth.sum())

        precision = precision_score(
            truth, prediction, zero_division=0
        )

        recall = recall_score(
            truth, prediction, zero_division=0
        )

        class_f1 = f1_score(
            truth, prediction, zero_division=0
        )

        # AP is undefined if this test subset has no positives.
        ap = (
            float(average_precision_score(truth, scores))
            if support > 0
            else float("nan")
        )

        if np.isfinite(ap):
            aps.append(ap)

        if "class_name" in class_df.columns:
            class_name = str(
                class_df.loc[class_id, "class_name"]
            )
        elif "label_name" in class_df.columns:
            class_name = str(
                class_df.loc[class_id, "label_name"]
            )
        else:
            class_name = f"class_{class_id}"

        per_class_rows.append({
            "experiment": experiment_name,
            "class_id": class_id,
            "class_name": class_name,
            "test_positive_support": support,
            "precision": precision,
            "recall": recall,
            "f1": class_f1,
            "average_precision": ap,
        })

    mean_ap = float(np.mean(aps)) if aps else float("nan")

    result = {
        "experiment": experiment_name,
        "test_locations": len(Y_test),
        "test_loss": test_loss,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "mAP": mean_ap,
        "classes_in_mAP": len(aps),
        "threshold": 0.5,
    }

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test Macro-F1: {macro_f1:.4f}")
    print(f"Test Micro-F1: {micro_f1:.4f}")
    print(f"Test mAP: {mean_ap:.4f}")
    print(f"Classes in mAP: {len(aps)}/{NUM_CLASSES}")

    return result, per_class_rows


# ============================================================
# 11. TRAIN BOTH EXPERIMENTS
# ============================================================

single_classifier, single_validation = train_classifier(
    X_train_single,
    y_train,
    X_val_single,
    y_val,
    "Single-date baseline",
    CHECKPOINT_DIR / "single_date_best.pt"
)

four_classifier, four_validation = train_classifier(
    X_train_four,
    y_train,
    X_val_four,
    y_val,
    "Four-date feature averaging",
    CHECKPOINT_DIR / "four_date_best.pt"
)


# ============================================================
# 12. SAVE VALIDATION RESULTS
# ============================================================

validation_results = pd.DataFrame([
    single_validation,
    four_validation,
])

validation_path = DATA_DIR / "full_data_validation_comparison.csv"

validation_results.to_csv(
    validation_path, index=False
)

print("\nValidation results saved:")
print(validation_path)


# ============================================================
# 13. FINAL TEST EVALUATION
# ============================================================

# Do not use these test results to select epochs or tune thresholds.
single_test, single_class_rows = evaluate_test(
    single_classifier,
    X_test_single,
    y_test,
    "Single-date baseline"
)

four_test, four_class_rows = evaluate_test(
    four_classifier,
    X_test_four,
    y_test,
    "Four-date feature averaging"
)


# ============================================================
# 14. SAVE OVERALL AND PER-CLASS TEST RESULTS
# ============================================================

test_results = pd.DataFrame([
    single_test,
    four_test,
])

test_path = DATA_DIR / "full_data_test_comparison.csv"

test_results.to_csv(
    test_path, index=False
)

per_class_results = pd.DataFrame(
    single_class_rows + four_class_rows
)

per_class_path = DATA_DIR / "full_data_per_class_test_metrics.csv"

per_class_results.to_csv(
    per_class_path, index=False
)

print("\n" + "=" * 72)
print("FULL-DATA EXPERIMENT COMPLETED")
print("=" * 72)

print("\nValidation comparison:")
print(validation_results.to_string(index=False))

print("\nHeld-out test comparison:")
print(test_results.to_string(index=False))

print("\nSaved files:")
print("Validation:", validation_path)
print("Test comparison:", test_path)
print("Per-class test metrics:", per_class_path)
print("Checkpoints:", CHECKPOINT_DIR)
print("Feature caches:", CACHE_DIR)

print("\nThe test split was not used for training or model selection.")
