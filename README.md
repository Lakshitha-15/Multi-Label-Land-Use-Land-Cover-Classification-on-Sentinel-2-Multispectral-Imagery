# Multi-Temporal LULC Classification — BigEarthNet-v2

This repository currently documents the **data processing and verification stage** of a multi-temporal land-use/land-cover classification project using BigEarthNet-v2 Sentinel-2 imagery.

The goal of this stage is to verify that the dataset contains multiple observations of the same geographic patch over time and to construct a consistent multi-temporal spectral representation for later model development.

## Current progress

- Analyzed BigEarthNet-v2 metadata.
- Extracted Sentinel-2 acquisition dates and tile identifiers.
- Created spatial patch IDs independent of acquisition date.
- Identified spatial locations with multiple observations.
- Checked whether spatial locations appear across multiple dataset splits.
- Selected a known four-observation spatial sequence.
- Verified the target IDs against the Hugging Face BigEarthNet-v2 dataset.
- Located the corresponding records in the local Parquet cache.
- Inspected the stored RGB, NIR and SWIR image fields.
- Decoded image bytes into NumPy arrays.
- Resized SWIR bands to a common `120 × 120` spatial resolution.
- Constructed a six-channel representation for each temporal observation.
- Constructed the final temporal tensor with shape:

```text
(4, 6, 120, 120)
```

## Spectral representation

The six channels are ordered as:

| Channel | Sentinel-2 band | Description |
|---|---|---|
| 0 | B02 | Blue |
| 1 | B03 | Green |
| 2 | B04 | Red |
| 3 | B08 | NIR |
| 4 | B11 | SWIR1 |
| 5 | B12 | SWIR2 |

Thus, the current representation is:

```text
4 dates × 6 bands × 120 × 120 pixels
```

## Repository structure

```text
BigEarthNet-Temporal-LULC/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── docs/
│   └── data_processing.md
└── scripts/
    ├── 01_metadata_analysis.py
    ├── 02_temporal_split_verification.py
    ├── 03_temporal_sequence_analysis.py
    ├── 04_huggingface_dataset_verification.py
    ├── 05_huggingface_cache_inspection.py
    ├── 06_target_sample_search.py
    └── 07_temporal_tensor_construction.py
```

## Setup

Create a virtual environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Place the BigEarthNet-v2 metadata file at:

```text
data/metadata.parquet
```

The actual image dataset is not stored in this repository. The processing scripts use the Hugging Face dataset/cache:

```text
timm/bigearthnet-v2-rgb-nir-swir
```

If the Hugging Face cache is in a non-standard location, set:

```bash
HF_HUB_CACHE=/path/to/huggingface/hub
```

On Windows PowerShell:

```powershell
$env:HF_HUB_CACHE="C:\path\to\huggingface\hub"
```

## Running the processing stages

Run the scripts in order:

```bash
python scripts/01_metadata_analysis.py
python scripts/02_temporal_split_verification.py
python scripts/03_temporal_sequence_analysis.py
python scripts/04_huggingface_dataset_verification.py
python scripts/05_huggingface_cache_inspection.py
python scripts/06_target_sample_search.py
python scripts/07_temporal_tensor_construction.py
```

The scripts are intentionally kept as separate stages because they represent the progression of the data investigation and verification work completed so far.

## Dataset

This project uses BigEarthNet-v2 and the corresponding Hugging Face RGB-NIR-SWIR representation. Dataset files are not included in this repository.

## Project status

**Current milestone: Data Processing & Verification complete.**

The repository currently stops at temporal tensor construction. Model development and classification experiments are intentionally not included yet.
