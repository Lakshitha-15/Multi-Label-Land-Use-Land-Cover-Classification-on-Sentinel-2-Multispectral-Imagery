
from pathlib import Path
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

TRAIN_LIMIT = 1024
VALIDATION_LIMIT = 512
BATCH_SIZE = 1
EPOCHS = 30
LEARNING_RATE = 1e-3
SEED = 42
NUM_CLASSES = 19

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")

print("=" * 70)
print("SINGLE-DATE VS FOUR-DATE LULC PILOT")
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
# LOAD DATASETS
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
val_generator = torch.Generator().manual_seed(SEED)

train_indices = torch.randperm(
    len(train_dataset), generator=train_generator
)[:TRAIN_LIMIT].tolist()

val_indices = torch.randperm(
    len(val_dataset), generator=val_generator
)[:VALIDATION_LIMIT].tolist()

train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(val_dataset, val_indices)

print("Full training set:", len(train_dataset))
print("Full validation set:", len(val_dataset))
print("Pilot training subset:", len(train_subset))
print("Pilot validation subset:", len(val_subset))


# ------------------------------------------------------------
# LOAD PRETRAINED FEATURE EXTRACTOR
# ------------------------------------------------------------

print("\nLoading pretrained ResNet50...")

model = model_module.TemporalResNet50(
    num_classes=NUM_CLASSES,
    pretrained=True
).to(device)

model.eval()

for parameter in model.parameters():
    parameter.requires_grad = False


# ------------------------------------------------------------
# EXTRACT FEATURES
#
# Each sequence produces:
#   1. Single-date feature: first observation
#   2. Four-date feature: mean of all four observations
#
# Each image is processed only once.
# ------------------------------------------------------------

def extract_features(dataset, split_name):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    single_features = []
    four_date_features = []
    all_targets = []

    print(f"\nExtracting {split_name} features...")

    with torch.no_grad():
        for batch_number, (images, targets) in enumerate(loader, 1):
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

            first_date = features[:, 0, :]
            four_dates = features.mean(dim=1)

            single_features.append(first_date.cpu())
            four_date_features.append(four_dates.cpu())
            all_targets.append(targets.cpu())

            if batch_number % 32 == 0:
                print(
                    f"  Processed {batch_number}/"
                    f"{len(loader)} sequences"
                )

    X_single = torch.cat(single_features, dim=0)
    X_four = torch.cat(four_date_features, dim=0)
    Y = torch.cat(all_targets, dim=0)

    print("Single-date features:", tuple(X_single.shape))
    print("Four-date features:", tuple(X_four.shape))
    print("Targets:", tuple(Y.shape))

    return X_single, X_four, Y


X_train_single, X_train_four, y_train = extract_features(
    train_subset, "training"
)

X_val_single, X_val_four, y_val = extract_features(
    val_subset, "validation"
)


# ------------------------------------------------------------
# CALCULATE CLASS WEIGHTS FROM THE FULL TRAINING SET
# ------------------------------------------------------------

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

print("\nClass weights loaded.")


# ------------------------------------------------------------
# TRAIN A LINEAR CLASSIFIER ON CACHED FEATURES
# ------------------------------------------------------------

def train_classifier(X_train, Y_train, X_val, Y_val, name):
    print("\n" + "-" * 65)
    print("Training:", name)
    print("-" * 65)

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

    for epoch in range(1, EPOCHS + 1):
        classifier.train()

        optimizer.zero_grad()

        logits = classifier(X_train.to(device))
        loss = criterion(logits, Y_train.to(device))

        assert torch.isfinite(loss), "Non-finite training loss!"

        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            classifier.eval()

            with torch.no_grad():
                val_logits = classifier(X_val.to(device))
                val_loss = criterion(
                    val_logits, Y_val.to(device)
                ).item()

            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"Train loss: {loss.item():.4f} | "
                f"Validation loss: {val_loss:.4f}"
            )

    classifier.eval()

    with torch.no_grad():
        probabilities = torch.sigmoid(
            classifier(X_val.to(device))
        ).cpu().numpy()

    y_true = Y_val.numpy().astype(int)
    y_pred = (probabilities >= 0.5).astype(int)

    macro_f1 = f1_score(
        y_true, y_pred,
        average="macro",
        zero_division=0
    )

    micro_f1 = f1_score(
        y_true, y_pred,
        average="micro",
        zero_division=0
    )

    # Compute AP only for classes with positive validation examples.
    valid_class_aps = []

    for class_id in range(NUM_CLASSES):
        if y_true[:, class_id].sum() > 0:
            ap = average_precision_score(
                y_true[:, class_id],
                probabilities[:, class_id]
            )
            valid_class_aps.append(ap)

    mean_ap = (
        float(np.mean(valid_class_aps))
        if valid_class_aps else float("nan")
    )

    print(f"\n{name} validation results:")
    print(f"  Macro-F1: {macro_f1:.4f}")
    print(f"  Micro-F1: {micro_f1:.4f}")
    print(f"  mAP:      {mean_ap:.4f}")
    print(
        f"  Classes included in mAP: "
        f"{len(valid_class_aps)}/{NUM_CLASSES}"
    )

    return {
        "experiment": name,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "mAP": mean_ap,
        "validation_loss": val_loss,
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
    "Four-date temporal model"
)


# ------------------------------------------------------------
# SAVE AND DISPLAY RESULTS
# ------------------------------------------------------------

results = pd.DataFrame([
    single_result,
    four_result
])

output_path = DATA_DIR / "pilot_single_vs_four_dates.csv"
results.to_csv(output_path, index=False)

print("\n" + "=" * 70)
print("PILOT COMPARISON")
print("=" * 70)
print(results.to_string(index=False))
print("\nSaved results to:", output_path)

print(
    "\nNOTE: These are preliminary validation results from a small "
    "subset with a frozen pretrained feature extractor."
)
print("The test split was not used.")
