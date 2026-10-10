# Multi-Temporal LULC Classification — BigEarthNet-v2

This repository contains the data preparation pipeline for a multi-temporal, multi-label Land Use/Land Cover (LULC) classification project using BigEarthNet-v2 Sentinel-2 multispectral imagery.

The project investigates whether **multiple temporal observations of the same geographic location improve multi-label land-cover classification compared with a single observation**.

The current work focuses on dataset analysis, temporal sequence construction, multispectral preprocessing, label validation, and training-only normalization. Model development and evaluation are the next stages.

## Current Progress

### Completed

- Analyzed BigEarthNet-v2 metadata and identified Sentinel-2 acquisition dates and spatial patch identifiers.
- Investigated temporal observation counts across training, validation, and test splits.
- Verified that spatial patches do not overlap across the original dataset splits.
- Identified locations with four observations on distinct acquisition dates.
- Located corresponding records in the local Hugging Face Parquet cache.
- Inspected and decoded RGB, NIR, and SWIR image fields.
- Reconstructed six-band multispectral observations at a common spatial resolution of `120 × 120`.
- Constructed and inspected sample temporal tensors.
- Identified and validated the 19-class label vocabulary.
- Built a full four-date temporal manifest containing Parquet shard and row references.
- Verified label consistency across all eligible four-date sequences.
- Constructed 19-dimensional multi-hot target vectors for multi-label classification.
- Computed per-band normalization statistics using training observations only.

### Dataset Summary

| Property | Value |
|---|---:|
| Dataset | BigEarthNet-v2 |
| Total indexed observations | 480,038 |
| Parquet shards | 64 |
| Land-cover classes | 19 |
| Spectral bands per observation | 6 |
| Temporal observations per sequence | 4 |
| Eligible spatial locations | 30,011 |
| Eligible temporal observations | 120,044 |

### Four-Date Dataset Cohort

| Split | Spatial locations | Observations |
|---|---:|---:|
| Training | 14,812 | 59,248 |
| Validation | 7,703 | 30,812 |
| Test | 7,496 | 29,984 |
| **Total** | **30,011** | **120,044** |

Each eligible location has four observations acquired on distinct dates. The spatial split assignments are retained from the dataset to prevent locations from appearing in multiple splits.

## Multispectral Representation

Each observation contains six spectral channels:

| Channel | Sentinel-2 band | Description |
|---|---|---|
| 0 | B02 | Blue |
| 1 | B03 | Green |
| 2 | B04 | Red |
| 3 | B08 | Near-infrared (NIR) |
| 4 | B11 | Short-wave infrared 1 (SWIR1) |
| 5 | B12 | Short-wave infrared 2 (SWIR2) |

The RGB, NIR, and SWIR fields are decoded and aligned to a common spatial resolution. SWIR bands are resized from `60 × 60` to `120 × 120` pixels using bilinear interpolation.

A single observation has the shape:

```text
(6, 120, 120)
```

A four-date temporal sequence has the shape:

```text
(4, 6, 120, 120)
```

The temporal dimension represents four acquisition dates, while the channel dimension represents the six spectral bands.

## Label Representation

The task is formulated as **19-class multi-label classification**. Each spatial location is represented by a 19-element multi-hot target vector, where `1` indicates the presence of a class and `0` indicates its absence.

Labels were checked across all four observations for every eligible location. No label inconsistencies were found in the four-date cohort.

The labels represent land-cover categories associated with the spatial patch; the project does not treat them as date-specific change annotations.

## Normalization

Per-band means and standard deviations were calculated from the training observations only.

| Band | Training mean | Training standard deviation |
|---|---:|---:|
| B02 | 99.2979 | 27.8932 |
| B03 | 120.8488 | 28.8200 |
| B04 | 120.7154 | 39.2936 |
| B08 | 2493.7702 | 1216.5429 |
| B11 | 2177.0961 | 1079.0137 |
| B12 | 1463.2236 | 873.9540 |

Normalization uses:

```text
normalized_value = (value - training_mean) / training_std
```

The same training statistics must be applied to training, validation, and test observations to prevent information leakage.

## Repository Structure

The main project structure is:

```text
BigEarthNet-Temporal-LULC/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── docs/
│   └── data_processing.md
├── scripts/
│   ├── 01_metadata_analysis.py
│   ├── 02_temporal_split_verification.py
│   ├── 03_temporal_sequence_analysis.py
│   ├── 04_huggingface_dataset_verification.py
│   ├── 05_huggingface_cache_inspection.py
│   ├── 06_target_sample_search.py
│   ├── 07_temporal_tensor_construction.py
│   ├── 08_temporal_dataset_builder.py
│   ├── 09_build_temporal_manifest.py
│   ├── 10_locate_sample_images.py
│   ├── 11_validate_multispectral_images.py
│   ├── 12_build_sample_tensors.py
│   ├── 13_inspect_sample_tensors.py
│   ├── 14_inspect_label_vocabulary.py
│   ├── 15_build_full_temporal_manifest.py
│   └── 16_compute_band_statistics.py
└── temporal_data/
    ├── train_index.csv
    ├── validation_index.csv
    ├── test_index.csv
    ├── sample_image_lookup.csv
    ├── label_vocabulary.csv
    ├── temporal_manifest.csv
    ├── temporal_sequences.csv
    ├── temporal_manifest_report.json
    ├── normalization_stats.json
    └── band_statistics.csv
```

The structure above illustrates the main files. The exact list of preprocessing scripts may grow as development continues.

## Setup

Use Python and install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

The data pipeline uses libraries including NumPy, pandas, PyArrow, Pillow, and PyTorch. Ensure the dependencies required by the scripts you run are installed in your active Python environment.

## Dataset Access

The project uses BigEarthNet-v2 through the Hugging Face dataset:

```text
timm/bigearthnet-v2-rgb-nir-swir
```

The full image dataset is not included in this repository. Processing scripts read the cached Parquet shards directly to avoid unnecessary dataset duplication and excessive disk usage.

The current preprocessing scripts use local paths configured in the scripts. Update these paths to match your environment before running them on another machine.

## Research and Model Development

The planned experimental workflow includes:

1. Building a memory-efficient PyTorch Dataset and DataLoader.
2. Validating batch shapes, normalization, and multi-label targets.
3. Training a single-date baseline model.
4. Training a four-date temporal classification model.
5. Comparing performance using macro-F1, micro-F1, mean average precision (mAP), and per-class metrics.
6. Investigating spectral-band and temporal contributions through ablation experiments.
7. Exploring model interpretability using Grad-CAM or an appropriate alternative.
8. Developing a prediction API and an interactive Streamlit interface.
9. Tracking experiments and preparing reproducible deployment and documentation.

These are planned stages; results will be documented as they are implemented and evaluated.

## Project Status

**Current milestone: Temporal Dataset Preparation and Validation Completed.**

The full four-date cohort has been indexed, its labels validated, and its training-only band statistics computed. The next milestone is implementing and validating the lazy PyTorch data pipeline, followed by baseline and temporal model training.

Classification performance and the benefit of temporal observations have not yet been established.
