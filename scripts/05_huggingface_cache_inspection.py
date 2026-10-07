"""Locate locally cached BigEarthNet-v2 Parquet shards without hard-coded user paths."""
import os
from pathlib import Path

DATASET_CACHE_NAME = "datasets--timm--bigearthnet-v2-rgb-nir-swir"


def find_cache_data_dirs():
    roots = []
    for env_name in ("HF_HUB_CACHE", "HF_HOME"):
        value = os.environ.get(env_name)
        if value:
            root = Path(value)
            if env_name == "HF_HOME":
                root = root / "hub"
            roots.append(root)

    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub" / "datasets")

    matches = []
    for root in roots:
        dataset_root = root / DATASET_CACHE_NAME
        if dataset_root.exists():
            matches.extend(dataset_root.glob("snapshots/*/data"))

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(matches))


cache_dirs = find_cache_data_dirs()

if not cache_dirs:
    print("BigEarthNet-v2 Hugging Face cache was not found.")
    print("Set HF_HUB_CACHE if your Hugging Face cache is stored elsewhere.")
else:
    for cache in cache_dirs:
        print("Cache:", cache)
        for file in sorted(cache.glob("test-*.parquet")):
            print(f"  {file.name}: {file.stat().st_size / (1024**2):.2f} MB")
