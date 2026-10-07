"""Extract four temporal observations and construct a 4 x 6 x 120 x 120 tensor."""
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image

DATASET_CACHE_NAME = "datasets--timm--bigearthnet-v2-rgb-nir-swir"
TARGET_IDS = [
    "S2A_MSIL2A_20170803T094031_N9999_R036_T34TCR_45_52",
    "S2A_MSIL2A_20171002T094031_N9999_R036_T34TCR_45_52",
    "S2B_MSIL2A_20171206T094349_N9999_R036_T34TCR_45_52",
    "S2A_MSIL2A_20180430T094031_N9999_R036_T34TCR_45_52",
]
TARGET_IDS_SET = set(TARGET_IDS)


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


def decode_image(image_data):
    return np.array(Image.open(BytesIO(image_data["bytes"])))


def resize_band(band, size=(120, 120)):
    tensor = torch.tensor(band, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return tensor.squeeze(0).squeeze(0).numpy()


cache = find_cache_data_dir()
if cache is None:
    raise FileNotFoundError(
        "BigEarthNet-v2 Hugging Face cache was not found. "
        "Set HF_HUB_CACHE if necessary."
    )

rows = []
columns = [
    "image_id", "image", "nir", "swir1", "swir2", "labels", "label_names",
    "country", "tile", "acquisition_time", "satellite", "relative_orbit",
    "patch_row", "patch_col", "epsg", "utm_x", "utm_y", "lat", "lon",
    "s1_name", "s2v1_name",
]

for file in sorted(cache.glob("test-*.parquet")):
    print("Reading:", file.name)
    parquet_file = pq.ParquetFile(file)

    for batch in parquet_file.iter_batches(columns=columns, batch_size=1000):
        batch_df = batch.to_pandas()
        matched = batch_df[batch_df["image_id"].isin(TARGET_IDS_SET)]
        if len(matched) > 0:
            rows.append(matched)
            print("  FOUND", len(matched), "target row(s)")

target_df = pd.concat(rows, ignore_index=True)
target_df["acquisition_time"] = pd.to_datetime(target_df["acquisition_time"])
target_df = target_df.sort_values("acquisition_time").reset_index(drop=True)

print("\n======================================")
print("TEMPORAL OBSERVATIONS")
print("======================================")
print(target_df[["image_id", "acquisition_time", "satellite", "country", "tile", "labels", "label_names"]].to_string(index=False))
print("\nNumber of observations:", len(target_df))

print("\n======================================")
print("IMAGE DATA INSPECTION")
print("======================================")


def inspect_image(image_data, name):
    print(f"\n{name}:")
    print("  Stored type:", type(image_data))
    if isinstance(image_data, dict):
        print("  Dictionary keys:", image_data.keys())
        if image_data.get("bytes") is not None:
            image = Image.open(BytesIO(image_data["bytes"]))
            print("  Decoded type:", type(image))
            print("  Size:", image.size)
            print("  Mode:", image.mode)
        elif image_data.get("path") is not None:
            print("  Path:", image_data["path"])
        else:
            print("  WARNING: No image bytes or path found.")
    else:
        print("  WARNING: Unexpected image format.")

for i, row in target_df.iterrows():
    print("\n--------------------------------------")
    print("Observation", i + 1)
    print("--------------------------------------")
    print("Date:", row["acquisition_time"])
    print("ID:", row["image_id"])
    inspect_image(row["image"], "RGB")
    inspect_image(row["nir"], "NIR")
    inspect_image(row["swir1"], "SWIR1")
    inspect_image(row["swir2"], "SWIR2")

print("\n======================================")
print("NUMPY ARRAY INSPECTION")
print("======================================")

for i, row in target_df.iterrows():
    rgb = decode_image(row["image"])
    nir = decode_image(row["nir"])
    swir1 = decode_image(row["swir1"])
    swir2 = decode_image(row["swir2"])

    print("\n--------------------------------------")
    print("Observation", i + 1)
    print("Date:", row["acquisition_time"])
    print("--------------------------------------")
    for name, array in [("RGB", rgb), ("NIR", nir), ("SWIR1", swir1), ("SWIR2", swir2)]:
        print(name)
        print("  Shape:", array.shape)
        print("  Dtype:", array.dtype)
        print("  Min:", array.min())
        print("  Max:", array.max())

# ------------------------------------------------------------
# Build one 6-channel observation for each date.
# Channel order: B02, B03, B04, B08, B11, B12
# ------------------------------------------------------------
temporal_images = []

for _, row in target_df.iterrows():
    rgb = decode_image(row["image"])
    nir = decode_image(row["nir"])
    swir1 = resize_band(decode_image(row["swir1"]))
    swir2 = resize_band(decode_image(row["swir2"]))

    b04 = rgb[:, :, 0].astype(np.float32)
    b03 = rgb[:, :, 1].astype(np.float32)
    b02 = rgb[:, :, 2].astype(np.float32)
    nir = nir.astype(np.float32)

    observation = np.stack([b02, b03, b04, nir, swir1, swir2], axis=0)
    temporal_images.append(observation)

temporal_tensor = np.stack(temporal_images, axis=0)

print("\n======================================")
print("TEMPORAL TENSOR")
print("======================================")
print("Shape:", temporal_tensor.shape)
print("Dtype:", temporal_tensor.dtype)
print("\nExpected:")
print("(4, 6, 120, 120)")
print("\nBand order:")
print("Channel 0 → B02 Blue")
print("Channel 1 → B03 Green")
print("Channel 2 → B04 Red")
print("Channel 3 → B08 NIR")
print("Channel 4 → B11 SWIR1")
print("Channel 5 → B12 SWIR2")

for t in range(len(target_df)):
    print("\nObservation", t + 1)
    print("Date:", target_df.iloc[t]["acquisition_time"])
    for c, band_name in enumerate(["B02", "B03", "B04", "B08", "B11", "B12"]):
        band = temporal_tensor[t, c]
        print(f"  {band_name}: shape = {band.shape}, min = {band.min()}, max = {band.max()}")
