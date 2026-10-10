
from pathlib import Path
from io import BytesIO
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image



# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

PROJECT_DIR = Path(
    r"D:\Documents\Sunanda College\Fifth Semester"
    r"\ML Lab\Project\BigEarthNet-Temporal-LULC-GitReady"
    r"\BigEarthNet-Temporal-LULC"
)

# Folder containing project-generated temporal data
DATA_DIR = PROJECT_DIR / "temporal_data"

# CSV containing the selected sample image locations
LOOKUP_FILE = DATA_DIR / "sample_image_lookup.csv"

# Hugging Face dataset cache (leave this unchanged)
CACHE_DIR = (
    Path(r"C:\Users\Admin\.cache\huggingface\hub")
    / "datasets--timm--bigearthnet-v2-rgb-nir-swir"
    / "snapshots"
    / "ec220eee66af0f15b37fb66c3205e381ea1bad51"
    / "data"
)

# Output folder for sample tensors and metadata
OUTPUT_DIR = DATA_DIR / "sample_tensors"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



# ------------------------------------------------------------
# IMAGE DECODING
# ------------------------------------------------------------

def decode_image(value):

    if value is None:
        raise ValueError("Missing image band.")

    if isinstance(value, dict):
        data = value.get("bytes")

        if data is None:
            raise ValueError("Image has no encoded bytes.")

        with Image.open(BytesIO(data)) as img:
            return img.copy()

    if isinstance(value, bytes):
        with Image.open(BytesIO(value)) as img:
            return img.copy()

    raise TypeError(f"Unexpected image type: {type(value)}")


def resize_numeric_band(image, size=(120, 120)):

    # Convert to floating point before interpolation so
    # fractional interpolated values are retained.
    array = np.asarray(image).astype(np.float32)

    float_image = Image.fromarray(array, mode="F")

    resized = float_image.resize(
        size,
        resample=Image.Resampling.BILINEAR
    )

    return np.asarray(resized, dtype=np.float32).copy()


def image_to_six_bands(row):

    # RGB image channels are stored as R, G, B.
    # Reverse them to obtain B, G, R order:
    # B02, B03, B04.
    rgb = np.asarray(
        decode_image(row["image"]).convert("RGB")
    ).astype(np.float32)

    blue_green_red = rgb[:, :, ::-1].copy()

    nir = np.asarray(
        decode_image(row["nir"])
    ).astype(np.float32)

    swir1 = resize_numeric_band(
        decode_image(row["swir1"])
    )

    swir2 = resize_numeric_band(
        decode_image(row["swir2"])
    )

    if blue_green_red.shape != (120, 120, 3):
        raise ValueError(
            f"Unexpected RGB shape: {blue_green_red.shape}"
        )

    if nir.shape != (120, 120):
        raise ValueError(f"Unexpected NIR shape: {nir.shape}")

    # Stack channels last, then transpose to C x H x W.
    bands_hwc = np.dstack([
        blue_green_red[:, :, 0],  # B02 - blue
        blue_green_red[:, :, 1],  # B03 - green
        blue_green_red[:, :, 2],  # B04 - red
        nir,                      # B08 - near infrared
        swir1,                    # B11 - shortwave infrared
        swir2                     # B12 - shortwave infrared
    ])

    return np.transpose(
        bands_hwc, (2, 0, 1)
    ).astype(np.float32)


# ------------------------------------------------------------
# READ ONLY THE REQUIRED ROW GROUPS
# ------------------------------------------------------------

def extract_selected_images(lookup):

    image_rows = {}
    grouped = lookup.groupby("shard", sort=True)

    columns = [
        "image_id",
        "image",
        "nir",
        "swir1",
        "swir2",
        "labels",
        "label_names"
    ]

    for shard_name, shard_group in grouped:

        shard_path = CACHE_DIR / shard_name

        print(f"\nReading selected rows from {shard_name}")

        parquet_file = pq.ParquetFile(shard_path)

        target_rows = {
            int(row.row_index): row.image_id
            for row in shard_group.itertuples()
        }

        cumulative_rows = 0

        for rg_index in range(parquet_file.num_row_groups):

            group_size = (
                parquet_file.metadata
                .row_group(rg_index)
                .num_rows
            )

            start = cumulative_rows
            end = start + group_size

            matching_rows = [
                index for index in target_rows
                if start <= index < end
            ]

            cumulative_rows = end

            if not matching_rows:
                continue

            table = parquet_file.read_row_group(
                rg_index,
                columns=columns
            )

            rows = table.to_pylist()

            for absolute_index in matching_rows:

                local_index = absolute_index - start
                row = rows[local_index]
                image_id = target_rows[absolute_index]

                if row["image_id"] != image_id:
                    raise ValueError(
                        "Row mismatch! Expected "
                        f"{image_id}, got {row['image_id']}"
                    )

                tensor = image_to_six_bands(row)

                image_rows[image_id] = {
                    "tensor": tensor,
                    "labels": row["labels"],
                    "label_names": row["label_names"]
                }

    return image_rows


# ------------------------------------------------------------
# BUILD FOUR-DATE SEQUENCES
# ------------------------------------------------------------

if __name__ == "__main__":

    lookup = pd.read_csv(LOOKUP_FILE)

    lookup["date"] = pd.to_datetime(lookup["date"])

    if lookup["image_id"].duplicated().any():
        raise ValueError("Duplicate image IDs in lookup.")

    image_rows = extract_selected_images(lookup)

    if len(image_rows) != len(lookup):
        raise ValueError(
            f"Expected {len(lookup)} images, "
            f"extracted {len(image_rows)}."
        )

    sequences = []
    metadata = []
    inconsistent_labels = []

    for (split, spatial_id), group in lookup.groupby(
        ["split", "spatial_id"], sort=True
    ):

        group = group.sort_values("date")

        if len(group) != 4:
            raise ValueError(
                f"{spatial_id} has {len(group)} observations."
            )

        sequence = np.stack([
            image_rows[image_id]["tensor"]
            for image_id in group["image_id"]
        ])

        if sequence.shape != (4, 6, 120, 120):
            raise ValueError(
                f"Unexpected sequence shape: {sequence.shape}"
            )

        labels_by_date = [
            image_rows[image_id]["labels"]
            for image_id in group["image_id"]
        ]

        names_by_date = [
            image_rows[image_id]["label_names"]
            for image_id in group["image_id"]
        ]

        # Preserve the source labels. Do not assume consistency.
        canonical_labels = [
            json.dumps(labels, sort_keys=True, default=str)
            for labels in labels_by_date
        ]

        labels_consistent = len(set(canonical_labels)) == 1

        if not labels_consistent:
            inconsistent_labels.append({
                "split": split,
                "spatial_id": spatial_id,
                "labels_by_date": labels_by_date
            })

        sequences.append(sequence)

        metadata.append({
            "split": split,
            "spatial_id": spatial_id,
            "dates": [
                date.strftime("%Y-%m-%d")
                for date in group["date"]
            ],
            "image_ids": group["image_id"].tolist(),
            "labels_by_date": labels_by_date,
            "label_names_by_date": names_by_date,
            "labels_consistent": labels_consistent
        })

    X = np.stack(sequences).astype(np.float32)

    tensor_path = OUTPUT_DIR / "sample_X.npy"
    metadata_path = OUTPUT_DIR / "sample_metadata.json"

    np.save(tensor_path, X)

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=str)

    print("\n" + "=" * 55)
    print("SAMPLE TEMPORAL TENSORS CREATED")
    print("=" * 55)

    print("Tensor shape:", X.shape)
    print("Data type:", X.dtype)
    print("All values finite:", bool(np.isfinite(X).all()))
    print("Minimum value:", float(X.min()))
    print("Maximum value:", float(X.max()))

    print("\nSequences by split:")
    print(
        pd.Series([item["split"] for item in metadata])
        .value_counts()
        .to_string()
    )

    print("\nLocations with inconsistent labels:",
          len(inconsistent_labels))

    if inconsistent_labels:
        print("First inconsistent example:")
        print(json.dumps(
            inconsistent_labels[0],
            indent=2,
            default=str
        ))

    print("\nSaved tensor:", tensor_path)
    print("Saved metadata:", metadata_path)
