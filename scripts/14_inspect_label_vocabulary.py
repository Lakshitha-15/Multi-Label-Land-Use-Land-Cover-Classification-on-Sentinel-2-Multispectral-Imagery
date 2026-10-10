
from pathlib import Path
from collections import Counter

import pyarrow.parquet as pq
import pandas as pd


PROJECT_DIR = Path(
    r"D:\Documents\Sunanda College\Fifth Semester"
    r"\ML Lab\Project\BigEarthNet-Temporal-LULC-GitReady"
    r"\BigEarthNet-Temporal-LULC"
)

CACHE_DIR = (
    Path(r"C:\Users\Admin\.cache\huggingface\hub")
    / "datasets--timm--bigearthnet-v2-rgb-nir-swir"
    / "snapshots"
    / "ec220eee66af0f15b37fb66c3205e381ea1bad51"
    / "data"
)

class_names_by_id = {}
class_counts = {
    "train": Counter(),
    "validation": Counter(),
    "test": Counter()
}

total_observations = Counter()

for split in ["train", "validation", "test"]:

    shard_paths = sorted(CACHE_DIR.glob(f"{split}-*.parquet"))

    print(f"\nScanning {split}: {len(shard_paths)} shards")

    for shard_path in shard_paths:

        parquet_file = pq.ParquetFile(shard_path)

        for rg_index in range(parquet_file.num_row_groups):

            table = parquet_file.read_row_group(
                rg_index,
                columns=["labels", "label_names"]
            )

            for row in table.to_pylist():

                labels = row["labels"]
                names = row["label_names"]

                total_observations[split] += 1

                if labels is None or names is None:
                    raise ValueError(
                        f"Missing labels in {shard_path.name}"
                    )

                if len(labels) != len(names):
                    raise ValueError(
                        f"Label/name mismatch in {shard_path.name}"
                    )

                for label_id, label_name in zip(labels, names):

                    label_id = int(label_id)
                    label_name = str(label_name)

                    previous_name = class_names_by_id.get(label_id)

                    if (
                        previous_name is not None
                        and previous_name != label_name
                    ):
                        raise ValueError(
                            f"Class ID {label_id} maps to both "
                            f"{previous_name} and {label_name}"
                        )

                    class_names_by_id[label_id] = label_name
                    class_counts[split][label_id] += 1

    print(
        f"Observations scanned: "
        f"{total_observations[split]:,}"
    )

print("\n" + "=" * 75)
print("CLASS VOCABULARY")
print("=" * 75)

rows = []

for label_id in sorted(class_names_by_id):

    rows.append({
        "class_id": label_id,
        "class_name": class_names_by_id[label_id],
        "train_count": class_counts["train"][label_id],
        "validation_count": class_counts["validation"][label_id],
        "test_count": class_counts["test"][label_id]
    })

vocabulary = pd.DataFrame(rows)

print(vocabulary.to_string(index=False))
print("\nNumber of classes:", len(vocabulary))
print("\nObservations:", dict(total_observations))

output_file = (
    PROJECT_DIR
    / "temporal_data"
    / "label_vocabulary.csv"
)

vocabulary.to_csv(output_file, index=False)

print("\nSaved vocabulary to:", output_file)
