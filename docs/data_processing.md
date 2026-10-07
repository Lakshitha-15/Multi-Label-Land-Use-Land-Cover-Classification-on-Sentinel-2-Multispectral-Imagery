# Data Processing & Verification

This milestone documents the work completed before model development.

## 1. Metadata analysis

`01_metadata_analysis.py` loads the BigEarthNet-v2 metadata and derives:

- Sentinel-2 acquisition date
- Sentinel-2 tile ID
- spatial patch ID independent of acquisition date
- number of observations for each spatial patch

The purpose is to identify geographic patches that have multiple Sentinel-2 observations over time.

## 2. Temporal split verification

`02_temporal_split_verification.py` examines the number of observations per spatial patch and checks whether the same spatial patch identifier occurs in more than one dataset split.

This is an important verification step because a temporal model should not accidentally receive observations of the same geographic location in both training and evaluation data.

## 3. Temporal sequence selection

`03_temporal_sequence_analysis.py` inspects the known spatial patch:

```text
T34TCR_45_52
```

Four observations were identified and ordered chronologically.

## 4. Dataset verification

`04_huggingface_dataset_verification.py` verifies that the metadata IDs correspond to records in:

```text
timm/bigearthnet-v2-rgb-nir-swir
```

The target observations are searched in the appropriate dataset split using their `image_id` values.

## 5. Local Parquet cache inspection

`05_huggingface_cache_inspection.py` locates the locally cached Hugging Face Parquet shards without relying on a machine-specific absolute path.

## 6. Target sample search

`06_target_sample_search.py` scans the cached Parquet shards in batches and finds the four target observations. Only the `image_id` column is read during this search to keep memory usage low.

## 7. Image decoding and temporal tensor construction

`07_temporal_tensor_construction.py` reads only the required target rows and performs the following steps:

1. Extract the stored RGB, NIR, SWIR1 and SWIR2 fields.
2. Decode the image bytes into NumPy arrays.
3. Split RGB into Sentinel-2 B04, B03 and B02 channels.
4. Resize SWIR1 and SWIR2 to `120 × 120`.
5. Convert the bands to `float32`.
6. Stack six channels in the order:

```text
B02, B03, B04, B08, B11, B12
```

7. Stack the four chronological observations.

The resulting representation is:

```text
(4, 6, 120, 120)
```

where the dimensions correspond to:

```text
4 temporal observations
6 spectral channels
120 × 120 spatial dimensions
```

This is the current endpoint of the data-processing milestone. Model architecture, training, evaluation and experiments will be added later.
