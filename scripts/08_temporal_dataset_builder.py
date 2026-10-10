
from pathlib import Path
import re

import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# 1. PATHS
# ============================================================

CACHE_DIR = (
    Path(r"C:\Users\Admin\.cache\huggingface\hub")
    / "datasets--timm--bigearthnet-v2-rgb-nir-swir"
    / "snapshots"
    / "ec220eee66af0f15b37fb66c3205e381ea1bad51"
    / "data"
)

OUTPUT_DIR = Path(
    r"D:\Documents\Sunanda College\Fifth Semester"
    r"\ML Lab\Project\temporal_data"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. EXTRACT DATE AND SPATIAL ID
# ============================================================

def parse_patch_id(image_id):

    date_match = re.search(
        r"MSIL2A_(\d{8})",
        image_id
    )

    spatial_match = re.search(
        r"(T\d{2}[A-Z]{3}_\d+_\d+)$",
        image_id
    )

    if date_match is None or spatial_match is None:
        return None

    date = pd.to_datetime(
        date_match.group(1),
        format="%Y%m%d"
    )

    spatial_id = spatial_match.group(1)

    return date, spatial_id


# ============================================================
# 3. BUILD INDEX FOR ONE SPLIT
# ============================================================

def build_split_index(split_name):

    files = sorted(
        CACHE_DIR.glob(f"{split_name}-*.parquet")
    )

    if not files:
        raise FileNotFoundError(
            f"No Parquet shards found for {split_name}"
        )

    records = []

    print(f"\n{'=' * 55}")
    print(f"PROCESSING {split_name.upper()}")
    print(f"Number of shards: {len(files)}")
    print("=" * 55)

    for file_number, file in enumerate(files, start=1):

        print(
            f"[{file_number}/{len(files)}] "
            f"Reading {file.name}"
        )

        parquet_file = pq.ParquetFile(file)

        for batch in parquet_file.iter_batches(
            columns=["image_id"],
            batch_size=10000
        ):

            image_ids = batch.column(
                "image_id"
            ).to_pylist()

            for image_id in image_ids:

                parsed = parse_patch_id(image_id)

                if parsed is None:
                    continue

                date, spatial_id = parsed

                records.append({
                    "image_id": image_id,
                    "spatial_id": spatial_id,
                    "date": date
                })

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError(
            f"No valid patch IDs found in {split_name}"
        )

    # Ensure an image ID is not accidentally duplicated.
    if df["image_id"].duplicated().any():
        raise ValueError(
            f"Duplicate image IDs found in {split_name}"
        )

    # Sort all observations chronologically per location.
    df = df.sort_values(
        ["spatial_id", "date"]
    ).reset_index(drop=True)

    # Count observations for each spatial location.
    df["num_observations"] = (
        df.groupby("spatial_id")["image_id"]
        .transform("count")
    )

    # Assign a chronological index: 1, 2, 3, ...
    df["time_index"] = (
        df.groupby("spatial_id").cumcount() + 1
    )

    # Save this split's index.
    output_file = (
        OUTPUT_DIR / f"{split_name}_index.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    # Print summary.
    unique_locations = df["spatial_id"].nunique()

    observation_counts = (
        df.groupby("spatial_id")["image_id"]
        .count()
        .value_counts()
        .sort_index()
    )

    print(f"\nTotal observations: {len(df):,}")
    print(f"Unique spatial patches: {unique_locations:,}")

    print("\nLocations by number of observations:")

    for count, locations in observation_counts.items():
        print(
            f"{count} observation(s): "
            f"{locations:,} location(s)"
        )

    print(f"\nSaved index: {output_file}")

    return df


# ============================================================
# 4. BUILD TRAIN, VALIDATION AND TEST INDEXES
# ============================================================

if __name__ == "__main__":

    train_df = build_split_index("train")

    validation_df = build_split_index("validation")

    test_df = build_split_index("test")

    print(f"\n{'=' * 55}")
    print("TEMPORAL INDEXING COMPLETE")
    print("=" * 55)

    print("Train rows:", f"{len(train_df):,}")
    print("Validation rows:", f"{len(validation_df):,}")
    print("Test rows:", f"{len(test_df):,}")

    print("\nOutput directory:")
    print(OUTPUT_DIR)
