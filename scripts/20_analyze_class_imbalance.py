
from pathlib import Path
import json

import pandas as pd


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "temporal_data"

SEQUENCES_FILE = DATA_DIR / "temporal_sequences.csv"
VOCABULARY_FILE = DATA_DIR / "label_vocabulary.csv"
OUTPUT_FILE = DATA_DIR / "class_distribution.csv"

SPLITS = ["train", "validation", "test"]
NUM_CLASSES = 19


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("=" * 65)
print("ANALYSING MULTI-LABEL CLASS DISTRIBUTION")
print("=" * 65)

sequences = pd.read_csv(SEQUENCES_FILE)

print("Total sequences:", len(sequences))

required = {"split", "target_vector_json"}

if not required.issubset(sequences.columns):
    raise ValueError(
        f"Missing required columns: {required - set(sequences.columns)}"
    )


# ------------------------------------------------------------
# LOAD CLASS NAMES
# ------------------------------------------------------------

vocabulary = pd.read_csv(VOCABULARY_FILE)

print("Vocabulary columns:", vocabulary.columns.tolist())

id_column = next(
    (c for c in ["class_id", "label_id", "id"] if c in vocabulary.columns),
    None
)

name_column = next(
    (c for c in ["class_name", "label_name", "name", "class"] 
     if c in vocabulary.columns),
    None
)

if id_column is None or name_column is None:
    raise ValueError(
        "Could not identify class ID/name columns in "
        f"{VOCABULARY_FILE.name}. Columns: {vocabulary.columns.tolist()}"
    )

class_names = {
    int(row[id_column]): str(row[name_column])
    for _, row in vocabulary.iterrows()
}

if len(class_names) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} classes, found {len(class_names)}"
    )


# ------------------------------------------------------------
# COUNT POSITIVE LABELS PER SEQUENCE
# ------------------------------------------------------------

counts = {
    split: [0] * NUM_CLASSES
    for split in SPLITS
}

sequence_totals = {
    split: 0
    for split in SPLITS
}

for row in sequences.itertuples(index=False):

    split = row.split

    if split not in counts:
        continue

    target = json.loads(row.target_vector_json)

    if len(target) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} target values, "
            f"found {len(target)}"
        )

    sequence_totals[split] += 1

    for class_id, value in enumerate(target):
        if value == 1:
            counts[split][class_id] += 1


# ------------------------------------------------------------
# BUILD OUTPUT TABLE
# ------------------------------------------------------------

records = []

for class_id in range(NUM_CLASSES):

    train_count = counts["train"][class_id]
    validation_count = counts["validation"][class_id]
    test_count = counts["test"][class_id]

    records.append({
        "class_id": class_id,
        "class_name": class_names[class_id],
        "train_count": train_count,
        "validation_count": validation_count,
        "test_count": test_count,
        "train_prevalence_percent": (
            100 * train_count / sequence_totals["train"]
        ),
        "candidate_pos_weight": (
            (sequence_totals["train"] - train_count) / train_count
            if train_count > 0 else None
        )
    })

result = pd.DataFrame(records)

result.to_csv(OUTPUT_FILE, index=False)


# ------------------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------------------

print("\nSEQUENCE COUNTS")
for split in SPLITS:
    print(f"{split}: {sequence_totals[split]:,}")

print("\nTRAINING CLASS FREQUENCIES")
print("-" * 65)

print(
    result[
        [
            "class_id",
            "class_name",
            "train_count",
            "train_prevalence_percent",
            "candidate_pos_weight"
        ]
    ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
)

print("\nSaved:", OUTPUT_FILE)
print("\nSUCCESS: Sequence-level class analysis completed.")
