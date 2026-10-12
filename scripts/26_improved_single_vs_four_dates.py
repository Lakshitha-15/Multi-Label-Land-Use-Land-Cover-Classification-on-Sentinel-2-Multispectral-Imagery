
from pathlib import Path
import copy
import importlib.util
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, average_precision_score


# ------------------------------------------------------------
# PATHS AND CONFIGURATION
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"
DATA_DIR = PROJECT_DIR / "temporal_data"
CACHE_DIR = DATA_DIR / "pilot_feature_cache"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_LIMIT = 4096
VALIDATION_LIMIT = 2048

BATCH_SIZE = 1
MAX_EPOCHS = 100
PATIENCE = 10
MIN_DELTA = 1e-4
LEARNING_RATE = 1e-3

SEED = 2026
NUM_CLASSES = 19

device = torch.device("cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 70)
print("IMPROVED SINGLE-DATE VS FOUR-DATE PILOT")
print("=" * 70)
print("Device:", device)


# ------------------------------------------------------------
# LOAD EXISTING SCRIPTS
# ------------------------------------------------------------

def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(
        module_name, file_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dataset_module = load_module(
    "temporal_dataset_module",
    SCRIPTS_DIR / "17_build_pytorch_dataset.py"
)

model_module = load_module(
    "pretrained_temporal_model_module",
    SCRIPTS_DIR / "23_test_pretrained_temporal_model.py"
)


# ------------------------------------------------------------
# LOAD DETERMINISTIC TRAIN / VALIDATION SUBSETS
# ------------------------------------------------------------

print("\nPreparing datasets...")

train_dataset = dataset_module.TemporalLULCDataset(
    split="train"
)

val_dataset = dataset_module.TemporalLULCDataset(
    split="validation"
)

assert len(train_dataset) >= TRAIN_LIMIT
assert len(val_dataset) >= VALIDATION_LIMIT

train_generator = torch.Generator().manual_seed(SEED)
val_generator = torch.Generator().manual_seed(SEED + 1)

train_indices = torch.randperm(
    len(train_dataset), generator=train_generator
)[:TRAIN_LIMIT].tolist()

val_indices = torch.randperm(
    len(val_dataset), generator=val_generator
)[:VALIDATION_LIMIT].tolist()

train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(val_dataset, val_indices)

print("Full training sequences:", len(train_dataset))
print("Full validation sequences:", len(val_dataset))
print("Pilot training locations:", len(train_subset))
print("Pilot validation locations:", len(val_subset))


# ------------------------------------------------------------
# FEATURE CACHE
# The cache filename records subset sizes and random seed.
# ------------------------------------------------------------

cache_path = CACHE_DIR / (
    f"features_train{TRAIN_LIMIT}_val{VALIDATION_LIMIT}"
    f"_seed{SEED}.npz"
)


# ------------------------------------------------------------
# LOAD PRETRAINED FEATURE EXTRACTOR
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# EXTRACT SINGLE-DATE AND FOUR-DATE FEATURES
# ------------------------------------------------------------

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

    print(f"\nExtracting {split_name} features...")

    with torch.no_grad():
        for batch_number, (images, targets) in enumerate(
            loader, start=1
        ):
            images = images.to(device)
            targets = targets.float().to(device)

            assert images.shape[1:] == (4, 6, 120, 120)

            batch_size, num_dates, num_bands, height, width = (
                images.shape
            )

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

            single_features.append(
                features[:, 0, :].cpu().numpy()
            )

            four_date_features.append(
                features.mean(dim=1).cpu().numpy()
            )

            targets_list.append(targets.cpu().numpy())

            if batch_number % 64 == 0:
                print(
                    f"  Processed {batch_number}/"
                    f"{len(loader)} sequences"
                )

    X_single = np.concatenate(single_features, axis=0)
    X_four = np.concatenate(four_date_features, axis=0)
    Y = np.concatenate(targets_list, axis=0)

    return X_single, X_four, Y


# ------------------------------------------------------------
# REUSE FEATURES IF THEY HAVE ALREADY BEEN EXTRACTED
# ------------------------------------------------------------

if cache_path.exists():
    print("\nFound feature cache:")
    print(cache_path)
    print("Loading cached features...")

    with np.load(cache_path, allow_pickle=False) as cache:
        X_train_single = cache["X_train_single"]
        X_train_four = cache["X_train_four"]
        y_train = cache["y_train"]

        X_val_single = cache["X_val_single"]
        X_val_four = cache["X_val_four"]
        y_val = cache["y_val"]

    expected_shapes = (
        (TRAIN_LIMIT, 2048),
        (TRAIN_LIMIT, 2048),
        (TRAIN_LIMIT, NUM_CLASSES),
        (VALIDATION_LIMIT, 2048),
        (VALIDATION_LIMIT, 2048),
        (VALIDATION_LIMIT, NUM_CLASSES)
    )

    actual_shapes = (
        X_train_single.shape,
        X_train_four.shape,
        y_train.shape,
        X_val_single.shape,
        X_val_four.shape,
        y_val.shape
    )

    if actual_shapes != expected_shapes:
        raise ValueError(
            "Cached feature shapes do not match this experiment. "
            "Remove the cache file and rerun the script."
        )

else:
    model = build_feature_extractor()

    X_train_single, X_train_four, y_train = extract_features(
        model, train_subset, "training"
    )

    X_val_single, X_val_four, y_val = extract_features(
        model, val_subset, "validation"
    )

    np.savez(
        cache_path,
        X_train_single=X_train_single,
        X_train_four=X_train_four,
        y_train=y_train,
        X_val_single=X_val_single,
        X_val_four=X_val_four,
        y_val=y_val
    )

    print("\nSaved extracted features:")
    print(cache_path)

print("\nFeature shapes:")
print("Train single-date:", X_train_single.shape)
print("Train four-date:", X_train_four.shape)
print("Validation:", y_val.shape)


# ------------------------------------------------------------
# VALIDATE LABELS AND LOAD FULL-TRAIN CLASS WEIGHTS
# ------------------------------------------------------------

assert np.isfinite(X_train_single).all()
assert np.isfinite(X_train_four).all()
assert np.isfinite(X_val_single).all()
assert np.isfinite(X_val_four).all()

assert np.isin(y_train, [0.0, 1.0]).all()
assert np.isin(y_val, [0.0, 1.0]).all()

class_df = pd.read_csv(
    DATA_DIR / "class_distribution.csv"
).sort_values("class_id")

assert class_df["class_id"].tolist() == list(range(NUM_CLASSES))

train_counts = torch.tensor(
    class_df["train_count"].to_numpy(),
    dtype=torch.float32
)

pos_weight = (
    (len(train_dataset) - train_counts) / train_counts
).clamp(min=1.0, max=20.0)

print("\nClass weights loaded from the full training split.")


# ------------------------------------------------------------
# TRAIN WITH EARLY STOPPING AND RESTORE BEST CHECKPOINT
# ------------------------------------------------------------

def train_classifier(
    X_train,
    Y_train,
    X_val,
    Y_val,
    experiment_name
):
    print("\n" + "-" * 70)
    print("Experiment:", experiment_name)
    print("-" * 70)

    # Give both classifiers the same initialization seed.
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
        X_train.shape[1], NUM_CLASSES
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
                f"Epoch {epoch:02d} | "
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
        raise RuntimeError("No valid model checkpoint was saved.")

    classifier.load_state_dict(best_state)
    classifier.eval()

    with torch.no_grad():
        probabilities = torch.sigmoid(
            classifier(X_val_t)
        ).cpu().numpy()

    y_true = Y_val.astype(int)
    y_pred = (probabilities >= 0.5).astype(int)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    micro_f1 = f1_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    valid_class_aps = []

    for class_id in range(NUM_CLASSES):
        if y_true[:, class_id].sum() > 0:
            valid_class_aps.append(
                average_precision_score(
                    y_true[:, class_id],
                    probabilities[:, class_id]
                )
            )

    mean_ap = (
        float(np.mean(valid_class_aps))
        if valid_class_aps
        else float("nan")
    )

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Micro-F1: {micro_f1:.4f}")
    print(f"mAP: {mean_ap:.4f}")
    print(
        f"Classes represented in mAP: "
        f"{len(valid_class_aps)}/{NUM_CLASSES}"
    )

    return {
        "experiment": experiment_name,
        "best_epoch": best_epoch,
        "epochs_trained": epochs_trained,
        "best_validation_loss": best_loss,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "mAP": mean_ap,
        "classes_in_mAP": len(valid_class_aps)
    }


# ------------------------------------------------------------
# RUN BOTH EXPERIMENTS
# ------------------------------------------------------------

single_result = train_classifier(
    X_train_single,
    y_train,
    X_val_single,
    y_val,
    "Single-date baseline"
)

four_result = train_classifier(
    X_train_four,
    y_train,
    X_val_four,
    y_val,
    "Four-date feature averaging"
)


# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

results = pd.DataFrame([
    single_result,
    four_result
])

output_path = DATA_DIR / "improved_pilot_comparison.csv"
results.to_csv(output_path, index=False)

print("\n" + "=" * 70)
print("IMPROVED PILOT COMPARISON")
print("=" * 70)
print(results.to_string(index=False))
print("\nResults saved to:", output_path)
print("Test split was not used.")
