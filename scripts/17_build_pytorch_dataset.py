
from pathlib import Path
from io import BytesIO
from collections import OrderedDict
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


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

SEQUENCE_FILE = DATA_DIR / "temporal_sequences.csv"
STATISTICS_FILE = DATA_DIR / "band_statistics.csv"

BAND_NAMES = ["B02", "B03", "B04", "B08", "B11", "B12"]

IMAGE_SIZE = (120, 120)
NUM_CLASSES = 19
NUM_DATES = 4

# Number of decoded Parquet row groups retained in memory.
ROW_GROUP_CACHE_SIZE = 2


# ------------------------------------------------------------
# IMAGE DECODING
# ------------------------------------------------------------

def decode_image(value):
    """Decode a stored image field into a NumPy array."""

    if isinstance(value, dict):
        # Support common image storage formats:
        # {"bytes": ...} or {"data": ...}
        value = value.get("bytes", value.get("data"))

    if isinstance(value, (bytes, bytearray, memoryview)):
        with Image.open(BytesIO(bytes(value))) as image:
            return np.array(image)

    if isinstance(value, Image.Image):
        return np.array(value)

    return np.asarray(value)


def decode_rgb(value):
    """Convert RGB storage order into B02, B03, B04 order."""

    array = decode_image(value)

    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(
            f"Unexpected RGB image shape: {array.shape}"
        )

    # Stored RGB channels represent B04, B03, B02.
    # Desired order is B02, B03, B04.
    array = array[:, :, :3][:, :, ::-1]

    return array.astype(np.float32)


def decode_single_band(value, band_name):
    """Decode a single spectral band."""

    array = decode_image(value)

    # Handle an image encoded with a singleton channel dimension.
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[:, :, 0]

    if array.ndim != 2:
        raise ValueError(
            f"{band_name} should be a 2D band, got {array.shape}"
        )

    array = array.astype(np.float32)

    if array.shape != IMAGE_SIZE:
        image = Image.fromarray(array)
        image = image.resize(
            IMAGE_SIZE,
            resample=Image.Resampling.BILINEAR,
        )
        array = np.asarray(image, dtype=np.float32)

    return array


def decode_observation(record):
    """Create a six-channel (6, 120, 120) observation."""

    blue, green, red = np.moveaxis(
        decode_rgb(record["image"]), -1, 0
    )

    nir = decode_single_band(record["nir"], "B08")
    swir1 = decode_single_band(record["swir1"], "B11")
    swir2 = decode_single_band(record["swir2"], "B12")

    observation = np.stack(
        [blue, green, red, nir, swir1, swir2],
        axis=0,
    )

    return observation.astype(np.float32)


# ------------------------------------------------------------
# LAZY TEMPORAL DATASET
# ------------------------------------------------------------

class TemporalLULCDataset(Dataset):

    def __init__(
        self,
        split,
        sequence_file=SEQUENCE_FILE,
        statistics_file=STATISTICS_FILE,
    ):
        self.split = split

        sequences = pd.read_csv(sequence_file)
        self.sequences = (
            sequences[sequences["split"] == split]
            .reset_index(drop=True)
        )

        if self.sequences.empty:
            raise ValueError(
                f"No sequences found for split: {split}"
            )

        # Read the exact training-only normalization statistics.
        stats = pd.read_csv(statistics_file)
        stats = stats.set_index("band")

        missing_bands = set(BAND_NAMES) - set(stats.index)

        if missing_bands:
            raise ValueError(
                f"Missing normalization statistics: {missing_bands}"
            )

        self.means = stats.loc[BAND_NAMES, "mean"].to_numpy(
            dtype=np.float32
        )

        self.stds = stats.loc[BAND_NAMES, "std"].to_numpy(
            dtype=np.float32
        )

        if not np.isfinite(self.means).all():
            raise ValueError("Non-finite normalization means.")

        if not np.isfinite(self.stds).all():
            raise ValueError("Non-finite normalization standard deviations.")

        if np.any(self.stds <= 0):
            raise ValueError("All band standard deviations must be positive.")

        self.means = self.means[:, None, None]
        self.stds = self.stds[:, None, None]

        # Each worker maintains its own small row-group cache.
        self.row_group_cache = OrderedDict()
        self.row_group_locations = {}

        # Map each shard's absolute row index to its row group.
        self.shard_paths = {
            path.name: path
            for path in CACHE_DIR.glob("*.parquet")
        }

        self.row_group_metadata = {}

        print(f"Preparing {split} split...")

        required_shards = set()

        for _, row in self.sequences.iterrows():
            required_shards.update(
                json.loads(row["shards_json"])
            )

        for shard_name in sorted(required_shards):
            if shard_name not in self.shard_paths:
                raise FileNotFoundError(
                    f"Shard not found: {shard_name}"
                )

            parquet = pq.ParquetFile(
                self.shard_paths[shard_name]
            )

            offsets = []
            current_offset = 0

            for group_index in range(parquet.num_row_groups):
                group_size = parquet.metadata.row_group(
                    group_index
                ).num_rows

                offsets.append({
                    "start": current_offset,
                    "end": current_offset + group_size,
                    "group_index": group_index,
                })

                current_offset += group_size

            self.row_group_metadata[shard_name] = offsets

        print(f"Sequences: {len(self.sequences):,}")
        print(f"Relevant shards: {len(required_shards)}")

    def __len__(self):
        return len(self.sequences)

    def _read_record(self, shard_name, row_index):
        """Read one record, caching recently used row groups."""

        row_index = int(row_index)
        cache_key = (shard_name, row_index)

        if cache_key in self.row_group_locations:
            group_key = self.row_group_locations[cache_key]
            table = self.row_group_cache[group_key]
            self.row_group_cache.move_to_end(group_key)

            row = table.slice(
                row_index - group_key[1], 1
            ).to_pylist()[0]

            return row

        # Locate the row group containing the absolute row index.
        group_info = None

        for info in self.row_group_metadata[shard_name]:
            if info["start"] <= row_index < info["end"]:
                group_info = info
                break

        if group_info is None:
            raise IndexError(
                f"Row {row_index} not found in {shard_name}"
            )

        group_index = group_info["group_index"]
        group_start = group_info["start"]
        group_key = (shard_name, group_start)

        if group_key not in self.row_group_cache:
            parquet = pq.ParquetFile(
                self.shard_paths[shard_name]
            )

            table = parquet.read_row_group(
                group_index,
                columns=[
                    "image_id",
                    "image",
                    "nir",
                    "swir1",
                    "swir2",
                ],
            )

            self.row_group_cache[group_key] = table

            # Remember where this row group begins.
            self.row_group_locations[
                (shard_name, group_start)
            ] = group_key

            while len(self.row_group_cache) > ROW_GROUP_CACHE_SIZE:
                evicted_key, _ = self.row_group_cache.popitem(
                    last=False
                )

                self.row_group_locations.pop(
                    evicted_key, None
                )

        table = self.row_group_cache[group_key]
        local_index = row_index - group_start

        actual_image_id = table.column("image_id")[
            local_index
        ].as_py()

        return table.slice(local_index, 1).to_pylist()[0]

    def __getitem__(self, index):
        row = self.sequences.iloc[index]

        image_ids = json.loads(row["image_ids_json"])
        shards = json.loads(row["shards_json"])
        row_indices = json.loads(row["row_indices_json"])
        target = json.loads(row["target_vector_json"])

        if not (
            len(image_ids)
            == len(shards)
            == len(row_indices)
            == NUM_DATES
        ):
            raise ValueError(
                f"Invalid temporal sequence at index {index}"
            )

        observations = []

        for expected_id, shard_name, row_index in zip(
            image_ids, shards, row_indices
        ):
            record = self._read_record(
                shard_name, row_index
            )

            if record["image_id"] != expected_id:
                raise ValueError(
                    "Image ID and Parquet row reference disagree: "
                    f"{expected_id} != {record['image_id']}"
                )

            observation = decode_observation(record)

            # Apply the same training statistics to every split.
            observation = (
                observation - self.means
            ) / self.stds

            observations.append(observation)

        # (4, 6, 120, 120)
        temporal_array = np.stack(observations, axis=0)

        # (19,)
        target_array = np.asarray(
            target, dtype=np.float32
        )

        if target_array.shape != (NUM_CLASSES,):
            raise ValueError(
                f"Invalid target shape: {target_array.shape}"
            )

        if not np.isfinite(temporal_array).all():
            raise ValueError(
                f"Non-finite image values at index {index}"
            )

        return (
            torch.from_numpy(
                np.ascontiguousarray(temporal_array)
            ),
            torch.from_numpy(target_array),
        )


# ------------------------------------------------------------
# TEST THE DATA PIPELINE
# ------------------------------------------------------------

def main():
    print("=" * 65)
    print("TESTING TEMPORAL PYTORCH DATASET")
    print("=" * 65)

    # Begin with two sequences to limit memory and runtime.
    dataset = TemporalLULCDataset(split="train")

    if len(dataset) < 2:
        raise RuntimeError("Not enough training sequences.")

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    images, targets = next(iter(loader))

    print("\nBATCH VALIDATION")
    print("-" * 45)

    print("Images shape:", tuple(images.shape))
    print("Targets shape:", tuple(targets.shape))
    print("Image dtype:", images.dtype)
    print("Target dtype:", targets.dtype)

    print("Images finite:", torch.isfinite(images).all().item())
    print("Targets finite:", torch.isfinite(targets).all().item())

    print("Image min:", images.min().item())
    print("Image max:", images.max().item())

    print("Labels per sample:", targets.sum(dim=1).tolist())

    assert images.shape == (2, 4, 6, 120, 120)
    assert targets.shape == (2, 19)
    assert torch.isfinite(images).all()
    assert torch.isfinite(targets).all()
    assert torch.all((targets == 0) | (targets == 1))

    print("\nSUCCESS: Dataset and batch checks passed.")


if __name__ == "__main__":
    main()
