
from pathlib import Path
import importlib.util
import random

import numpy as np
import torch
from torch.utils.data import DataLoader


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"
DATA_DIR = PROJECT_DIR / "temporal_data"

CACHE_DIR = DATA_DIR / "full_feature_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 1
SEED = 2026
NUM_CLASSES = 19
FEATURE_DIM = 2048
NUM_DATES = 4

EXPECTED_COUNTS = {
    "train": 14812,
    "validation": 7703,
    "test": 7496,
}

device = torch.device("cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 70)
print("EXTRACTING FOUR-DATE TEMPORAL FEATURES")
print("=" * 70)
print("Device:", device)


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
    "temporal_feature_dataset_module",
    SCRIPTS_DIR / "17_build_pytorch_dataset.py"
)

model_module = load_module(
    "temporal_feature_model_module",
    SCRIPTS_DIR / "23_test_pretrained_temporal_model.py"
)


# ============================================================
# 3. PREPARE DATASETS
# ============================================================

print("\nPreparing datasets...")

datasets = {
    "train": dataset_module.TemporalLULCDataset(split="train"),
    "validation": dataset_module.TemporalLULCDataset(
        split="validation"
    ),
    "test": dataset_module.TemporalLULCDataset(split="test"),
}

for split_name, dataset in datasets.items():
    actual_count = len(dataset)
    expected_count = EXPECTED_COUNTS[split_name]

    print(f"{split_name.capitalize()}: {actual_count:,} samples")

    if actual_count != expected_count:
        raise ValueError(
            f"Unexpected {split_name} count: {actual_count}; "
            f"expected {expected_count}."
        )


# ============================================================
# 4. LOAD FROZEN PRETRAINED RESNET50
# ============================================================

print("\nLoading pretrained ResNet50...")

model = model_module.TemporalResNet50(
    num_classes=NUM_CLASSES,
    pretrained=True
).to(device)

model.eval()

for parameter in model.parameters():
    parameter.requires_grad = False

print("Feature extractor loaded.")


# ============================================================
# 5. EXTRACT FEATURES FOR ONE SPLIT
# ============================================================

def extract_temporal_features(dataset, split_name):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    temporal_features = []
    targets_list = []

    total = len(loader)

    print(f"\nExtracting {split_name} features...")

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

            # Process each date with the same frozen backbone.
            flattened = images.reshape(
                batch_size * num_dates,
                num_bands,
                height,
                width
            )

            features = model.feature_extractor(flattened)

            # Restore the temporal dimension.
            features = features.reshape(
                batch_size, num_dates, -1
            )

            if tuple(features.shape[1:]) != (
                NUM_DATES, FEATURE_DIM
            ):
                raise ValueError(
                    f"Unexpected feature shape: {features.shape}"
                )

            # Preserve all four dates, in their existing order.
            temporal_features.append(
                features.cpu().numpy().astype(np.float32)
            )

            targets_list.append(
                targets.cpu().numpy().astype(np.float32)
            )

            if (
                batch_number % 256 == 0
                or batch_number == total
            ):
                print(
                    f"  Processed {batch_number:,}/{total:,}"
                )

    X_temporal = np.concatenate(temporal_features, axis=0)
    Y = np.concatenate(targets_list, axis=0)

    expected_feature_shape = (
        len(dataset), NUM_DATES, FEATURE_DIM
    )
    expected_target_shape = (len(dataset), NUM_CLASSES)

    if X_temporal.shape != expected_feature_shape:
        raise ValueError(
            f"Unexpected temporal feature shape: "
            f"{X_temporal.shape}"
        )

    if Y.shape != expected_target_shape:
        raise ValueError(
            f"Unexpected label shape: {Y.shape}"
        )

    if not np.isfinite(X_temporal).all():
        raise ValueError("Non-finite temporal features detected.")

    if not np.isfinite(Y).all():
        raise ValueError("Non-finite labels detected.")

    if not np.isin(Y, [0, 1]).all():
        raise ValueError("Labels must be binary.")

    return X_temporal, Y


# ============================================================
# 6. EXTRACT AND SAVE EACH SPLIT
# ============================================================

for split_name, dataset in datasets.items():

    output_path = CACHE_DIR / (
        f"temporal_features_{split_name}_seed{SEED}.npz"
    )

    if output_path.exists():
        print(f"\nExisting temporal cache found: {output_path.name}")
        print("Skipping extraction. Delete it only if you want to rebuild it.")
        continue

    X_temporal, Y = extract_temporal_features(
        dataset, split_name
    )

    temporary_path = output_path.with_suffix(".tmp.npz")

    np.savez_compressed(
        temporary_path,
        X_temporal=X_temporal,
        Y=Y
    )

    temporary_path.replace(output_path)

    print(f"\nSaved: {output_path}")
    print("Temporal features:", X_temporal.shape)
    print("Labels:", Y.shape)

    del X_temporal, Y


# ============================================================
# 7. FINISH
# ============================================================

del model

print("\n" + "=" * 70)
print("TEMPORAL FEATURE EXTRACTION COMPLETE")
print("=" * 70)
