
from pathlib import Path
from io import BytesIO

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

PROJECT_DIR = Path(
    r"D:\Documents\Sunanda College\Fifth Semester"
    r"\ML Lab\Project"
)

DATA_DIR = PROJECT_DIR / "temporal_data"


CACHE_DIR = (
    Path(r"C:\Users\Admin\.cache\huggingface\hub")
    / "datasets--timm--bigearthnet-v2-rgb-nir-swir"
    / "snapshots"
    / "ec220eee66af0f15b37fb66c3205e381ea1bad51"
    / "data"
)

print("Cache directory:", CACHE_DIR)
print("Directory exists:", CACHE_DIR.exists())


lookup = pd.read_csv(DATA_DIR / "sample_image_lookup.csv")

# Columns expected from the dataset.
band_columns = ["image", "nir", "swir1", "swir2"]


# ------------------------------------------------------------
# DECODE ONE IMAGE FIELD
# ------------------------------------------------------------

def decode_image(value):

    if value is None:
        return None

    # Hugging Face image fields are commonly dictionaries
    # containing encoded image bytes and an optional path.
    if isinstance(value, dict):
        image_bytes = value.get("bytes")

        if image_bytes is None:
            raise ValueError(
                "Image field has no encoded bytes."
            )

        return Image.open(BytesIO(image_bytes)).copy()

    # Also support a direct bytes value.
    if isinstance(value, bytes):
        return Image.open(BytesIO(value)).copy()

    raise TypeError(
        f"Unexpected image field type: {type(value)}"
    )


# ------------------------------------------------------------
# VALIDATE SELECTED ROWS
# ------------------------------------------------------------

def validate_shard(shard_name, target_rows):

    shard_path = CACHE_DIR / shard_name
    parquet_file = pq.ParquetFile(shard_path)

    # Read only columns that actually exist.
    available_columns = parquet_file.schema_arrow.names

    required_columns = ["image_id"] + [
        col for col in band_columns
        if col in available_columns
    ]

    if "image" not in available_columns:
        raise ValueError(
            f"'image' column not found in {shard_name}. "
            f"Available columns: {available_columns}"
        )

    print(f"\nShard: {shard_name}")
    print("Available columns:", available_columns)

    # Map target absolute row positions to row groups.
    row_group_targets = {}
    cumulative_rows = 0

    for rg_index in range(parquet_file.num_row_groups):

        rg_rows = parquet_file.metadata
        group_size = rg_rows.row_group(rg_index).num_rows

        start = cumulative_rows
        end = start + group_size

        matching = [
            row for row in target_rows
            if start <= row < end
        ]

        if matching:
            row_group_targets[rg_index] = matching

        cumulative_rows = end

    results = []

    for rg_index, matching_rows in row_group_targets.items():

        table = parquet_file.read_row_group(
            rg_index,
            columns=required_columns
        )

        rows = table.to_pylist()

        group_start = sum(
            parquet_file.metadata.row_group(i).num_rows
            for i in range(rg_index)
        )

        for absolute_row in matching_rows:

            row = rows[absolute_row - group_start]

            print("\nImage ID:", row["image_id"])
            print("Parquet row:", absolute_row)

            for band in required_columns:
                if band == "image_id":
                    continue

                img = decode_image(row[band])

                if img is None:
                    print(f"  {band}: MISSING")
                    continue

                print(
                    f"  {band}: size={img.size}, "
                    f"mode={img.mode}"
                )

                if band == "image":
                    if img.size != (120, 120):
                        print("  WARNING: unexpected RGB dimensions")

                if band in ("nir",):
                    if img.size != (120, 120):
                        print("  WARNING: unexpected NIR dimensions")

                if band in ("swir1", "swir2"):
                    if img.size != (60, 60):
                        print("  WARNING: unexpected SWIR dimensions")

                img.close()

            results.append(row["image_id"])

    return results


# ------------------------------------------------------------
# RUN VALIDATION
# ------------------------------------------------------------

if __name__ == "__main__":

    found_ids = []

    for shard_name, group in lookup.groupby("shard"):

        target_rows = group["row_index"].astype(int).tolist()

        found_ids.extend(
            validate_shard(shard_name, target_rows)
        )

    print("\n" + "=" * 55)
    print("MULTISPECTRAL VALIDATION SUMMARY")
    print("=" * 55)

    print("Expected images:", len(lookup))
    print("Decoded image records:", len(found_ids))
    print("Unique decoded IDs:", len(set(found_ids)))

    if len(found_ids) != len(lookup):
        print("WARNING: Some selected images were not decoded.")

    print("\nValidation complete.")
