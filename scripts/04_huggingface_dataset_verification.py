"""Verify metadata IDs against the Hugging Face BigEarthNet-v2 dataset."""
import pandas as pd
from datasets import load_dataset

METADATA_PATH = "data/metadata.parquet"
TARGET_SPATIAL_ID = "T34TCR_45_52"
DATASET_NAME = "timm/bigearthnet-v2-rgb-nir-swir"

a = pd.read_parquet(METADATA_PATH)
a["date"] = pd.to_datetime(
    a["patch_id"].str.extract(r"MSIL2A_(\d{8})")[0],
    format="%Y%m%d",
)
a["tile"] = a["patch_id"].str.extract(r"(T\d{2}[A-Z]{3})")
a["spatial_id"] = a["patch_id"].str.extract(
    r"(T\d{2}[A-Z]{3}_\d+_\d+)$"
)

print("Loading a sample from the Hugging Face dataset...")
hf_dataset = load_dataset(DATASET_NAME, split="train", streaming=False)
sample = next(iter(hf_dataset))

print("\nSample schema:")
print(sample.keys())
for key, value in sample.items():
    if hasattr(value, "shape"):
        print(key, value.shape)
    else:
        print(key, type(value), value)

target = a[a["spatial_id"] == TARGET_SPATIAL_ID].sort_values("date")
print("\nTarget metadata:")
print(target[["patch_id", "date", "split", "tile", "spatial_id", "labels"]].to_string(index=False))

target_ids = target["patch_id"].tolist()
target_split = target["split"].iloc[0]

print("\nTarget split:", target_split)
print("\nTarget IDs:")
for image_id in target_ids:
    print(image_id)

print("\nSearching the corresponding Hugging Face split...")
hf_dataset = load_dataset(DATASET_NAME, split=target_split, streaming=True)
found = {}

for sample in hf_dataset:
    image_id = sample["image_id"]
    if image_id in target_ids:
        found[image_id] = sample
        print("FOUND:", image_id)
        if len(found) == len(target_ids):
            break

print("\n-----------------------------")
print("Expected:", len(target_ids))
print("Found:", len(found))
print("-----------------------------")
