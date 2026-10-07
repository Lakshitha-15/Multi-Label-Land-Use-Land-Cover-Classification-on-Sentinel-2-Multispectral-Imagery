"""Find the four target observations inside the locally cached Parquet shards."""
import os
from pathlib import Path
import pyarrow.parquet as pq

DATASET_CACHE_NAME = "datasets--timm--bigearthnet-v2-rgb-nir-swir"
TARGET_IDS = {
    "S2A_MSIL2A_20170803T094031_N9999_R036_T34TCR_45_52",
    "S2A_MSIL2A_20171002T094031_N9999_R036_T34TCR_45_52",
    "S2B_MSIL2A_20171206T094349_N9999_R036_T34TCR_45_52",
    "S2A_MSIL2A_20180430T094031_N9999_R036_T34TCR_45_52",
}


def find_cache_data_dir():
    roots = []
    for env_name in ("HF_HUB_CACHE", "HF_HOME"):
        value = os.environ.get(env_name)
        if value:
            root = Path(value)
            if env_name == "HF_HOME":
                root = root / "hub"
            roots.append(root)
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    for root in roots:
        dataset_root = root / DATASET_CACHE_NAME
        matches = sorted(dataset_root.glob("snapshots/*/data"))
        if matches:
            return matches[-1]
    return None


cache = find_cache_data_dir()
if cache is None:
    raise FileNotFoundError(
        "BigEarthNet-v2 Hugging Face cache was not found. "
        "Set HF_HUB_CACHE if necessary."
    )

found = {}

for file in sorted(cache.glob("test-*.parquet")):
    print("Checking:", file.name)
    parquet_file = pq.ParquetFile(file)

    for batch in parquet_file.iter_batches(columns=["image_id"], batch_size=10000):
        image_ids = batch.column("image_id").to_pylist()
        for row_index, image_id in enumerate(image_ids):
            if image_id in TARGET_IDS:
                found[image_id] = (file, row_index)
                print("FOUND:", image_id)

print("\n==============================")
print("RESULT")
print("==============================")
for target_id in sorted(TARGET_IDS):
    if target_id in found:
        print("FOUND:", target_id)
        print("FILE :", found[target_id][0].name)
    else:
        print("NOT FOUND:", target_id)
print("\nTotal found:", len(found), "/", len(TARGET_IDS))
