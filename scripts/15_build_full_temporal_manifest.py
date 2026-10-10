
from pathlib import Path
from collections import Counter
import json
import re

import pandas as pd
import pyarrow.parquet as pq


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

OUTPUT_MANIFEST = DATA_DIR / "temporal_manifest.csv"
OUTPUT_SEQUENCES = DATA_DIR / "temporal_sequences.csv"
OUTPUT_REPORT = DATA_DIR / "temporal_manifest_report.json"
VOCABULARY_FILE = DATA_DIR / "label_vocabulary.csv"


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

EXPECTED_CLASSES = 19
EXPECTED_OBSERVATIONS = 4

SPLITS = ["train", "validation", "test"]

DATE_PATTERN = re.compile(r"MSIL2A_(\d{8})")
SPATIAL_PATTERN = re.compile(r"(T\d{2}[A-Z]{3}_\d+_\d+)$")


# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------

def parse_json_list(value):
    """Convert a CSV cell containing a JSON list into a Python list."""
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    if isinstance(value, str):
        return json.loads(value)

    raise ValueError(f"Unexpected list value: {value!r}")


def get_split_from_shard(shard_name):
    """Identify the split from a shard filename."""
    for split in SPLITS:
        if shard_name.startswith(split + "-"):
            return split

    raise ValueError(f"Unknown split in shard name: {shard_name}")


def get_date(image_id):
    match = DATE_PATTERN.search(image_id)

    if match is None:
        raise ValueError(f"Cannot extract acquisition date: {image_id}")

    return match.group(1)


def get_spatial_id(image_id):
    match = SPATIAL_PATTERN.search(image_id)

    if match is None:
        raise ValueError(f"Cannot extract spatial ID: {image_id}")

    return match.group(1)


# ------------------------------------------------------------
# CHECK INPUT FILES
# ------------------------------------------------------------

print("=" * 65)
print("BUILDING FULL FOUR-DATE TEMPORAL MANIFEST")
print("=" * 65)

DATA_DIR.mkdir(parents=True, exist_ok=True)

required_files = [
    DATA_DIR / "train_index.csv",
    DATA_DIR / "validation_index.csv",
    DATA_DIR / "test_index.csv",
    VOCABULARY_FILE,
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")

if not CACHE_DIR.exists():
    raise FileNotFoundError(f"Parquet cache not found: {CACHE_DIR}")


# ------------------------------------------------------------
# LOAD THE EXISTING TEMPORAL INDEXES
# ------------------------------------------------------------

index_frames = []

for split in SPLITS:
    index_path = DATA_DIR / f"{split}_index.csv"

    frame = pd.read_csv(index_path)
    frame["split"] = split

    required_columns = {
        "image_id",
        "spatial_id",
        "date",
        "num_observations",
    }

    missing = required_columns - set(frame.columns)

    if missing:
        raise ValueError(
            f"{index_path.name} is missing columns: {sorted(missing)}"
        )

    index_frames.append(frame)

index_df = pd.concat(index_frames, ignore_index=True)

if index_df["image_id"].duplicated().any():
    duplicates = index_df.loc[
        index_df["image_id"].duplicated(keep=False), "image_id"
    ].head(10).tolist()

    raise ValueError(f"Duplicate image IDs in temporal indexes: {duplicates}")

print(f"Indexed observations: {len(index_df):,}")


# ------------------------------------------------------------
# LOAD THE 19-CLASS VOCABULARY
# ------------------------------------------------------------

vocab_df = pd.read_csv(VOCABULARY_FILE)

if not {"class_id", "class_name"}.issubset(vocab_df.columns):
    raise ValueError(
        "label_vocabulary.csv must contain class_id and class_name."
    )

vocab_df["class_id"] = vocab_df["class_id"].astype(int)
vocab_df = vocab_df.sort_values("class_id").reset_index(drop=True)

class_ids = vocab_df["class_id"].tolist()

if len(vocab_df) != EXPECTED_CLASSES:
    raise ValueError(
        f"Expected {EXPECTED_CLASSES} classes, "
        f"found {len(vocab_df)}."
    )

if class_ids != list(range(EXPECTED_CLASSES)):
    raise ValueError(
        f"Expected class IDs 0 to {EXPECTED_CLASSES - 1}; "
        f"found {class_ids}."
    )

class_names = dict(
    zip(
        vocab_df["class_id"],
        vocab_df["class_name"],
    )
)

print(f"Loaded {len(class_names)} classes.")


# ------------------------------------------------------------
# SELECT FOUR-OBSERVATION LOCATIONS
# ------------------------------------------------------------

group_sizes = (
    index_df.groupby(["split", "spatial_id"])
    .size()
    .rename("actual_observations")
    .reset_index()
)

eligible_groups = group_sizes[
    group_sizes["actual_observations"] == EXPECTED_OBSERVATIONS
].copy()

eligible_keys = set(
    zip(eligible_groups["split"], eligible_groups["spatial_id"])
)

eligible_index = index_df[
    index_df.apply(
        lambda row: (row["split"], row["spatial_id"]) in eligible_keys,
        axis=1,
    )
].copy()

# Keep only groups with four distinct acquisition dates.
distinct_dates = (
    eligible_index.groupby(["split", "spatial_id"])["date"]
    .nunique()
)

valid_date_keys = set(
    distinct_dates[distinct_dates == EXPECTED_OBSERVATIONS].index
)

eligible_index = eligible_index[
    eligible_index.apply(
        lambda row: (row["split"], row["spatial_id"]) in valid_date_keys,
        axis=1,
    )
].copy()

eligible_ids = set(eligible_index["image_id"])

print("\nEligible four-date locations:")

eligible_counts = (
    eligible_index[["split", "spatial_id"]]
    .drop_duplicates()
    .groupby("split")
    .size()
)

for split in SPLITS:
    print(f"  {split}: {eligible_counts.get(split, 0):,}")

print(f"Total eligible observations: {len(eligible_index):,}")


# ------------------------------------------------------------
# SCAN PARQUET METADATA ONLY
# ------------------------------------------------------------

shard_files = sorted(CACHE_DIR.glob("*.parquet"))

if not shard_files:
    raise FileNotFoundError(
        f"No Parquet shards found directly inside: {CACHE_DIR}"
    )

print(f"\nParquet shards found: {len(shard_files)}")
print("Scanning image IDs and labels; image arrays will not be loaded.")

row_lookup = {}
found_ids = set()

for shard_number, shard_path in enumerate(shard_files, start=1):
    split = get_split_from_shard(shard_path.name)

    parquet_file = pq.ParquetFile(shard_path)
    absolute_row_offset = 0

    for row_group_index in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(
            row_group_index,
            columns=["image_id", "labels", "label_names"],
        )

        image_ids = table.column("image_id").to_pylist()
        labels_list = table.column("labels").to_pylist()
        names_list = table.column("label_names").to_pylist()

        for local_index, image_id in enumerate(image_ids):
            absolute_row_index = absolute_row_offset + local_index

            # Ignore observations that are not in the four-date cohort.
            if image_id not in eligible_ids:
                continue

            if image_id in found_ids:
                raise ValueError(
                    f"Image ID appears more than once: {image_id}"
                )

            labels = labels_list[local_index] or []
            label_names = names_list[local_index] or []

            labels = sorted({int(label) for label in labels})
            label_names = list(label_names)

            unknown_labels = set(labels) - set(class_names)

            if unknown_labels:
                raise ValueError(
                    f"Unknown label IDs for {image_id}: "
                    f"{sorted(unknown_labels)}"
                )

            row_lookup[image_id] = {
                "split": split,
                "shard": shard_path.name,
                "row_index": absolute_row_index,
                "labels": labels,
                "label_names": label_names,
            }

            found_ids.add(image_id)

        absolute_row_offset += len(image_ids)

    print(
        f"[{shard_number:02d}/{len(shard_files):02d}] "
        f"{shard_path.name} | "
        f"rows: {absolute_row_offset:,} | "
        f"eligible IDs located: {len(found_ids):,}"
    )


# ------------------------------------------------------------
# CHECK THAT ALL ELIGIBLE OBSERVATIONS WERE LOCATED
# ------------------------------------------------------------

missing_ids = eligible_ids - found_ids

if missing_ids:
    examples = sorted(missing_ids)[:10]

    raise RuntimeError(
        f"Could not locate {len(missing_ids):,} eligible image IDs. "
        f"Examples: {examples}"
    )

if len(found_ids) != len(eligible_ids):
    raise RuntimeError(
        "The number of located observations does not match the index."
    )

print(f"\nAll {len(found_ids):,} eligible observations located.")


# ------------------------------------------------------------
# JOIN INDEX AND PARQUET ROW REFERENCES
# ------------------------------------------------------------

lookup_df = pd.DataFrame.from_dict(
    row_lookup,
    orient="index",
)

lookup_df.index.name = "image_id"
lookup_df = lookup_df.reset_index()

manifest = eligible_index.merge(
    lookup_df,
    on="image_id",
    how="left",
    validate="one_to_one",
    suffixes=("_index", "_parquet"),
)

if manifest["shard"].isna().any():
    raise RuntimeError("Some manifest rows have no shard reference.")

if (manifest["split_index"] != manifest["split_parquet"]).any():
    raise RuntimeError("A split index disagrees with its Parquet shard.")

manifest["split"] = manifest["split_index"]
manifest["date"] = manifest["date"].astype(str).str.replace(
    r"\.0$", "", regex=True
)

# Recalculate these from image IDs so the grouping is explicit.
manifest["date"] = manifest["image_id"].map(get_date)
manifest["spatial_id"] = manifest["image_id"].map(get_spatial_id)

manifest = manifest.sort_values(
    ["split", "spatial_id", "date", "image_id"]
).reset_index(drop=True)

manifest["time_index"] = (
    manifest.groupby(["split", "spatial_id"]).cumcount()
)


# ------------------------------------------------------------
# VALIDATE EACH TEMPORAL SEQUENCE AND BUILD TARGETS
# ------------------------------------------------------------

manifest_records = []
sequence_records = []
label_mismatches = []
sequence_errors = []

for (split, spatial_id), group in manifest.groupby(
    ["split", "spatial_id"],
    sort=True,
):
    group = group.sort_values(["date", "image_id"]).reset_index(drop=True)

    if len(group) != EXPECTED_OBSERVATIONS:
        sequence_errors.append(
            f"{split}/{spatial_id}: found {len(group)} observations"
        )
        continue

    if group["date"].nunique() != EXPECTED_OBSERVATIONS:
        sequence_errors.append(
            f"{split}/{spatial_id}: dates are not all distinct"
        )
        continue

    labels_per_date = group["labels"].tolist()
    reference_labels = labels_per_date[0]

    if any(labels != reference_labels for labels in labels_per_date[1:]):
        label_mismatches.append({
            "split": split,
            "spatial_id": spatial_id,
            "dates": group["date"].tolist(),
            "labels_by_date": labels_per_date,
        })
        continue

    target = [0] * EXPECTED_CLASSES

    for label_id in reference_labels:
        target[label_id] = 1

    target_names = [
        class_names[label_id]
        for label_id in reference_labels
    ]

    for time_index, row in group.iterrows():
        manifest_records.append({
            "split": split,
            "spatial_id": spatial_id,
            "time_index": time_index,
            "date": row["date"],
            "image_id": row["image_id"],
            "shard": row["shard"],
            "row_index": int(row["row_index"]),
            "labels_json": json.dumps(reference_labels),
            "label_names_json": json.dumps(target_names),
            "target_vector_json": json.dumps(target),
        })

    sequence_records.append({
        "split": split,
        "spatial_id": spatial_id,
        "num_observations": EXPECTED_OBSERVATIONS,
        "dates_json": json.dumps(group["date"].tolist()),
        "image_ids_json": json.dumps(group["image_id"].tolist()),
        "shards_json": json.dumps(group["shard"].tolist()),
        "row_indices_json": json.dumps(
            [int(value) for value in group["row_index"].tolist()]
        ),
        "labels_json": json.dumps(reference_labels),
        "label_names_json": json.dumps(target_names),
        "target_vector_json": json.dumps(target),
        "num_labels": len(reference_labels),
    })


# ------------------------------------------------------------
# SAVE VALIDATION REPORT
# ------------------------------------------------------------

report = {
    "project": "Multi-Label Temporal LULC Classification",
    "expected_classes": EXPECTED_CLASSES,
    "expected_observations_per_sequence": EXPECTED_OBSERVATIONS,
    "parquet_shards_scanned": len(shard_files),
    "indexed_observations": int(len(index_df)),
    "eligible_observations": int(len(eligible_index)),
    "eligible_locations_by_split": {
        split: int(
            eligible_index.loc[
                eligible_index["split"] == split, "spatial_id"
            ].nunique()
        )
        for split in SPLITS
    },
    "located_observations": int(len(found_ids)),
    "valid_sequences": int(len(sequence_records)),
    "manifest_observations": int(len(manifest_records)),
    "label_mismatch_count": len(label_mismatches),
    "sequence_error_count": len(sequence_errors),
    "label_mismatch_examples": label_mismatches[:20],
    "sequence_error_examples": sequence_errors[:20],
    "status": "passed" if not label_mismatches and not sequence_errors else "failed",
}

with open(OUTPUT_REPORT, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=2, ensure_ascii=False)


# ------------------------------------------------------------
# STOP IF VALIDATION FAILED
# ------------------------------------------------------------

if label_mismatches or sequence_errors:
    print("\nVALIDATION FAILED.")

    print(f"Label mismatches: {len(label_mismatches)}")
    print(f"Sequence errors: {len(sequence_errors)}")
    print(f"Report saved to: {OUTPUT_REPORT}")

    if label_mismatches:
        print("\nFirst label mismatch:")
        print(json.dumps(label_mismatches[0], indent=2))

    if sequence_errors:
        print("\nFirst sequence error:")
        print(sequence_errors[0])

    raise RuntimeError(
        "Manifest validation failed. Inspect the report before training."
    )


# ------------------------------------------------------------
# SAVE FINAL MANIFESTS
# ------------------------------------------------------------

manifest_df = pd.DataFrame(manifest_records)
sequences_df = pd.DataFrame(sequence_records)

manifest_df.to_csv(OUTPUT_MANIFEST, index=False)
sequences_df.to_csv(OUTPUT_SEQUENCES, index=False)


# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("MANIFEST BUILD COMPLETED SUCCESSFULLY")
print("=" * 65)

print(f"Observation manifest: {OUTPUT_MANIFEST}")
print(f"Sequence manifest:    {OUTPUT_SEQUENCES}")
print(f"Validation report:    {OUTPUT_REPORT}")

print("\nValidated sequences by split:")

for split in SPLITS:
    count = int((sequences_df["split"] == split).sum())
    print(f"  {split}: {count:,}")

print(f"\nTotal sequences: {len(sequences_df):,}")
print(f"Total observations: {len(manifest_df):,}")
print(f"Number of classes: {EXPECTED_CLASSES}")
print("Label mismatches: 0")
print("Sequence errors: 0")
print("\nNext step: estimate per-band normalization statistics")
print("using the training split only.")
