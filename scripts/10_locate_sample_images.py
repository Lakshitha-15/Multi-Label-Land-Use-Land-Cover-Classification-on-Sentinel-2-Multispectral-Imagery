
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

PROJECT_DIR = Path(
    r"D:\Documents\Sunanda College\Fifth Semester"
    r"\ML Lab\Project"
)

CACHE_DIR = (
    Path(r"C:\Users\Admin\.cache\huggingface\hub")
    / "datasets--timm--bigearthnet-v2-rgb-nir-swir"
    / "snapshots"
    / "ec220eee66af0f15b37fb66c3205e381ea1bad51"
    / "data"
)

INDEX_DIR = PROJECT_DIR / "temporal_data"
OUTPUT_FILE = INDEX_DIR / "sample_image_lookup.csv"

SAMPLES_PER_SPLIT = 5


# ------------------------------------------------------------
# 1. SELECT SAMPLE FOUR-DATE LOCATIONS
# ------------------------------------------------------------

sample_frames = []
target_ids = set()

for split_name in ["train", "validation", "test"]:

    index_file = INDEX_DIR / f"{split_name}_index.csv"
    df = pd.read_csv(index_file)

    four_date = df[df["num_observations"] == 4].copy()

    # Choose five reproducible sample locations.
    selected_locations = (
        four_date["spatial_id"]
        .drop_duplicates()
        .sample(
            n=min(SAMPLES_PER_SPLIT,
                  four_date["spatial_id"].nunique()),
            random_state=42
        )
    )

    selected = four_date[
        four_date["spatial_id"].isin(selected_locations)
    ].copy()

    selected["split"] = split_name
    selected["date"] = pd.to_datetime(selected["date"])

    sample_frames.append(selected)
    target_ids.update(selected["image_id"].tolist())

samples = pd.concat(sample_frames, ignore_index=True)

print("Selected locations:", samples.groupby("split")["spatial_id"].nunique().to_dict())
print("Target images:", len(target_ids))


# ------------------------------------------------------------
# 2. FIND EACH IMAGE IN THE PARQUET SHARDS
# ------------------------------------------------------------

lookup = {}
shard_files = sorted(CACHE_DIR.glob("*.parquet"))

for shard_number, shard_path in enumerate(shard_files, start=1):

    print(
        f"[{shard_number}/{len(shard_files)}] "
        f"Scanning {shard_path.name}"
    )

    parquet_file = pq.ParquetFile(shard_path)
    row_offset = 0

    # Read IDs only, not the large image payloads.
    for batch in parquet_file.iter_batches(
        columns=["image_id"],
        batch_size=10000
    ):

        image_ids = batch.column("image_id").to_pylist()

        for local_index, image_id in enumerate(image_ids):

            if image_id in target_ids:
                lookup[image_id] = {
                    "shard": shard_path.name,
                    "row_index": row_offset + local_index
                }

        row_offset += len(image_ids)

    if len(lookup) == len(target_ids):
        print("All target images located.")
        break


# ------------------------------------------------------------
# 3. SAVE LOOKUP
# ------------------------------------------------------------

samples["shard"] = samples["image_id"].map(
    lambda image_id: lookup.get(image_id, {}).get("shard")
)

samples["row_index"] = samples["image_id"].map(
    lambda image_id: lookup.get(image_id, {}).get("row_index")
)

missing = samples[samples["shard"].isna()]

if not missing.empty:
    print("\nWARNING: Some image IDs were not found:")
    print(missing[["split", "image_id"]].to_string(index=False))

samples.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 55)
print("IMAGE LOOKUP COMPLETE")
print("=" * 55)
print("Located images:", samples["shard"].notna().sum())
print("Missing images:", len(missing))
print("Saved to:", OUTPUT_FILE)

print("\nImages by split:")
print(samples.groupby("split")["shard"].apply(
    lambda values: values.notna().sum()
))
