
from pathlib import Path
import numpy as np

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

CACHE_DIR = PROJECT_DIR / "temporal_data" / "full_feature_cache"

CACHE_FILES = {
    "train": CACHE_DIR / "full_features_train_seed2026.npz",
    "validation": CACHE_DIR / "full_features_validation_seed2026.npz",
    "test": CACHE_DIR / "full_features_test_seed2026.npz",
}


# ------------------------------------------------------------
# INSPECT CACHED ARRAYS
# ------------------------------------------------------------

print("=" * 65)
print("INSPECTING FULL-DATA FEATURE CACHES")
print("=" * 65)

for split, file_path in CACHE_FILES.items():

    print(f"\n{'-' * 65}")
    print(f"Split: {split.upper()}")
    print(f"File: {file_path.name}")

    if not file_path.exists():
        print("ERROR: Cache file not found!")
        continue

    file_size_mb = file_path.stat().st_size / (1024 ** 2)
    print(f"File size: {file_size_mb:.2f} MB")

    with np.load(file_path, allow_pickle=False) as data:

        print("\nArrays stored in this file:")

        for key in data.files:
            array = data[key]

            print(f"\n  Name: {key}")
            print(f"  Shape: {array.shape}")
            print(f"  Data type: {array.dtype}")

            if np.issubdtype(array.dtype, np.number):
                print(f"  All finite: {np.isfinite(array).all()}")

                if array.size > 0:
                    print(f"  Minimum: {array.min():.6f}")
                    print(f"  Maximum: {array.max():.6f}")

            # Inspect likely label arrays without assuming their names.
            if "label" in key.lower() or "target" in key.lower():
                if array.size > 0:
                    print(f"  Positive entries: {np.count_nonzero(array > 0)}")

print("\n" + "=" * 65)
print("CACHE INSPECTION COMPLETE")
print("=" * 65)
