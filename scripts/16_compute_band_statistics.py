
from pathlib import Path
from io import BytesIO
import json
import math

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
import torch
import torch.nn.functional as F


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

PROJECT_DIR = Path(
    r"D:\Documents\Sunanda College\Fifth Semester"
    r"\ML Lab\Project\BigEarthNet-Temporal-LULC-GitReady"
    r"\BigEarthNet-Temporal-LULC"
)

DATA_DIR = PROJECT_DIR / "temporal_data"

CACHE_DIR = (
    Path(r"C:\Users\Admin\.cache\huggingface\hub")
    / "datasets--timm--bigearthnet-v2-rgb-nir-swir"
    / "snapshots"
    / "ec220eee66af0f15b37fb66c3205e381ea1bad51"
    / "data"
)

MANIFEST_FILE = DATA_DIR / "temporal_manifest.csv"
OUTPUT_JSON = DATA_DIR / "normalization_stats.json"
OUTPUT_CSV = DATA_DIR / "band_statistics.csv"


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

BAND_NAMES = ["B02", "B03", "B04", "B08", "B11", "B12"]
IMAGE_SIZE = (120, 120)

IMAGE_COLUMNS = ["image", "nir", "swir1", "swir2"]


# ------------------------------------------------------------
# DECODE AN IMAGE FIELD
# ------------------------------------------------------------

def decode_raster(value):
    """
    Decode a Hugging Face image field or an array-like value.
    Returns a NumPy array.
    """
    if isinstance(value, dict):
        image_bytes = value.get("bytes")

        if image_bytes is None:
            raise ValueError(
                "Image field has no embedded bytes. "
                "Check the Parquet image representation."
            )

        with Image.open(BytesIO(image_bytes)) as img:
            img.load()
            array = np.array(img)

    elif isinstance(value, np.ndarray):
        array = value

    else:
        array = np.asarray(value)

    return array


# ------------------------------------------------------------
# BUILD THE SIX-BAND IMAGE
# ------------------------------------------------------------

def build_six_band_image(row):
    """
    Returns float32 array with shape (6, 120, 120).

    Band order:
    B02, B03, B04, B08, B11, B12
    """

    rgb = decode_raster(row["image"])

    if rgb.shape != (120, 120, 3):
        raise ValueError(
            f"Unexpected RGB image shape: {rgb.shape}"
        )

    # Stored RGB order is B04, B03, B02.
    b04 = rgb[:, :, 0].astype(np.float32)
    b03 = rgb[:, :, 1].astype(np.float32)
    b02 = rgb[:, :, 2].astype(np.float32)

    b08 = decode_raster(row["nir"]).astype(np.float32)
    b11 = decode_raster(row["swir1"]).astype(np.float32)
    b12 = decode_raster(row["swir2"]).astype(np.float32)

    # Some decoders may return a singleton channel dimension.
    if b08.ndim == 3 and b08.shape[-1] == 1:
        b08 = b08[:, :, 0]

    if b11.ndim == 3 and b11.shape[-1] == 1:
        b11 = b11[:, :, 0]

    if b12.ndim == 3 and b12.shape[-1] == 1:
        b12 = b12[:, :, 0]

    if b08.shape != IMAGE_SIZE:
        raise ValueError(f"Unexpected B08 shape: {b08.shape}")

    # Resize SWIR bands using the same bilinear approach
    # used in the sample tensor builder.
    def resize_swir(band):
        if band.shape == IMAGE_SIZE:
            return band

        tensor = torch.from_numpy(band.copy())[None, None, :, :]

        resized = F.interpolate(
            tensor,
            size=IMAGE_SIZE,
            mode="bilinear",
            align_corners=False,
        )

        return resized[0, 0].numpy().astype(np.float32)

    b11 = resize_swir(b11)
    b12 = resize_swir(b12)

    bands = np.stack(
        [b02, b03, b04, b08, b11, b12],
        axis=0,
    ).astype(np.float32)

    if bands.shape != (6, 120, 120):
        raise ValueError(f"Unexpected final tensor shape: {bands.shape}")

    if not np.isfinite(bands).all():
        raise ValueError("Non-finite values found in an image.")

    return bands


# ------------------------------------------------------------
# LOAD TRAINING OBSERVATION REFERENCES
# ------------------------------------------------------------

print("=" * 65)
print("COMPUTING TRAINING-ONLY BAND STATISTICS")
print("=" * 65)

if not MANIFEST_FILE.exists():
    raise FileNotFoundError(f"Manifest not found: {MANIFEST_FILE}")

if not CACHE_DIR.exists():
    raise FileNotFoundError(f"Parquet cache not found: {CACHE_DIR}")

manifest = pd.read_csv(MANIFEST_FILE)

required_columns = {"split", "shard", "row_index", "image_id"}
missing_columns = required_columns - set(manifest.columns)

if missing_columns:
    raise ValueError(
        f"Manifest is missing columns: {sorted(missing_columns)}"
    )

train_df = manifest[manifest["split"] == "train"].copy()

if train_df.empty:
    raise ValueError("No training observations found in the manifest.")

train_df["row_index"] = train_df["row_index"].astype(int)

if train_df.duplicated(["shard", "row_index"]).any():
    raise ValueError("Duplicate shard/row references found.")

print(f"Training observations: {len(train_df):,}")
print(f"Training locations: {train_df['spatial_id'].nunique():,}")


# ------------------------------------------------------------
# PREPARE ROW REFERENCES BY SHARD
# ------------------------------------------------------------

rows_by_shard = {}

for shard_name, group in train_df.groupby("shard"):
    rows_by_shard[shard_name] = sorted(
        group["row_index"].astype(int).tolist()
    )

shard_lookup = {path.name: path for path in CACHE_DIR.glob("*.parquet")}

missing_shards = set(rows_by_shard) - set(shard_lookup)

if missing_shards:
    raise FileNotFoundError(
        f"Missing Parquet shards: {sorted(missing_shards)}"
    )


# ------------------------------------------------------------
# RUNNING STATISTICS
# ------------------------------------------------------------

# Accumulate in float64 to reduce rounding error.
band_sums = np.zeros(6, dtype=np.float64)
band_squared_sums = np.zeros(6, dtype=np.float64)
pixel_counts = np.zeros(6, dtype=np.int64)

processed_observations = 0


# ------------------------------------------------------------
# READ ONLY ROW GROUPS CONTAINING TRAINING OBSERVATIONS
# ------------------------------------------------------------

shard_names = sorted(rows_by_shard)

for shard_number, shard_name in enumerate(shard_names, start=1):
    shard_path = shard_lookup[shard_name]
    wanted_rows = rows_by_shard[shard_name]
    wanted_position = 0

    parquet_file = pq.ParquetFile(shard_path)
    row_group_start = 0

    for row_group_index in range(parquet_file.num_row_groups):
        row_group_size = parquet_file.metadata.row_group(
            row_group_index
        ).num_rows

        row_group_end = row_group_start + row_group_size

        # Identify selected absolute row indexes in this row group.
        selected_rows = []

        while wanted_position < len(wanted_rows):
            row_index = wanted_rows[wanted_position]

            if row_index >= row_group_end:
                break

            if row_index < row_group_start:
                raise ValueError(
                    f"Row index ordering error in {shard_name}: "
                    f"{row_index}"
                )

            selected_rows.append(row_index - row_group_start)
            wanted_position += 1

        if selected_rows:
            table = parquet_file.read_row_group(
                row_group_index,
                columns=IMAGE_COLUMNS,
            )

            for local_index in selected_rows:
                row = {
                    column: table.column(column)[local_index].as_py()
                    for column in IMAGE_COLUMNS
                }

                bands = build_six_band_image(row)

                # Calculate each band's sums over all pixels.
                for band_index in range(6):
                    band = bands[band_index].astype(np.float64)

                    band_sums[band_index] += band.sum()
                    band_squared_sums[band_index] += np.square(
                        band
                    ).sum()
                    pixel_counts[band_index] += band.size

                processed_observations += 1

        row_group_start = row_group_end

    if wanted_position != len(wanted_rows):
        raise RuntimeError(
            f"Not all requested rows were processed in {shard_name}. "
            f"Processed {wanted_position} of {len(wanted_rows)}."
        )

    print(
        f"[{shard_number:02d}/{len(shard_names):02d}] "
        f"{shard_name} | "
        f"observations processed: {processed_observations:,}"
    )


# ------------------------------------------------------------
# CALCULATE MEAN AND POPULATION STANDARD DEVIATION
# ------------------------------------------------------------

if processed_observations != len(train_df):
    raise RuntimeError(
        f"Expected {len(train_df):,} observations, "
        f"processed {processed_observations:,}."
    )

means = band_sums / pixel_counts

variances = (
    band_squared_sums / pixel_counts
) - np.square(means)

# Guard against tiny negative values due to floating-point rounding.
variances = np.maximum(variances, 0.0)

standard_deviations = np.sqrt(variances)

if not np.isfinite(means).all():
    raise ValueError("Non-finite band means.")

if not np.isfinite(standard_deviations).all():
    raise ValueError("Non-finite band standard deviations.")

if np.any(standard_deviations == 0):
    raise ValueError("At least one band has zero standard deviation.")


# ------------------------------------------------------------
# SAVE STATISTICS
# ------------------------------------------------------------

statistics = {
    "dataset": "BigEarthNet-v2",
    "split_used": "train",
    "observations_used": int(processed_observations),
    "locations_used": int(train_df["spatial_id"].nunique()),
    "image_height": IMAGE_SIZE[0],
    "image_width": IMAGE_SIZE[1],
    "band_order": BAND_NAMES,
    "rgb_band_note": (
        "B02, B03, B04 originate from the stored 8-bit tone-mapped RGB image."
    ),
    "swir_resize": (
        "B11 and B12 resized to 120x120 using bilinear interpolation, "
        "align_corners=False."
    ),
    "statistic": "population mean and population standard deviation",
    "bands": {},
}

for i, band_name in enumerate(BAND_NAMES):
    statistics["bands"][band_name] = {
        "mean": float(means[i]),
        "std": float(standard_deviations[i]),
        "pixel_count": int(pixel_counts[i]),
    }

with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
    json.dump(statistics, file, indent=2)

statistics_df = pd.DataFrame({
    "band": BAND_NAMES,
    "mean": means,
    "std": standard_deviations,
    "pixel_count": pixel_counts,
})

statistics_df.to_csv(OUTPUT_CSV, index=False)


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("BAND STATISTICS COMPLETED")
print("=" * 65)

print(f"Observations processed: {processed_observations:,}")
print(f"Locations used: {train_df['spatial_id'].nunique():,}")

print("\nNormalization statistics:")
print(statistics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

print(f"\nJSON saved to: {OUTPUT_JSON}")
print(f"CSV saved to:  {OUTPUT_CSV}")

print("\nUse these same training statistics for train, validation, and test.")
