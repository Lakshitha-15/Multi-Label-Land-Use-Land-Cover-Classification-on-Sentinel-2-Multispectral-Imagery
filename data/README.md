# Data

The repository does **not** include the BigEarthNet-v2 dataset or Hugging Face cache.

Place the metadata file used by the processing scripts at:

```text
data/metadata.parquet
```

The image data is expected to be available through the Hugging Face dataset/cache:

```text
timm/bigearthnet-v2-rgb-nir-swir
```

The scripts locate the Hugging Face cache from the standard local cache location. If the cache is stored elsewhere, set `HF_HUB_CACHE` before running the scripts.

The actual dataset files should remain outside Git because they are large and are not required to demonstrate the processing workflow.
