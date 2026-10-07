"""Verify temporal observations and check for spatial IDs crossing dataset splits."""
import pandas as pd

METADATA_PATH = "data/metadata.parquet"
TARGET_SPATIAL_ID = "T34TCR_45_52"

df = pd.read_parquet(METADATA_PATH)
df["date"] = pd.to_datetime(
    df["patch_id"].str.extract(r"MSIL2A_(\d{8})")[0],
    format="%Y%m%d",
)
df["spatial_id"] = df["patch_id"].str.extract(
    r"(T\d{2}[A-Z]{3}_\d+_\d+)$"
)

df["num_observations"] = df["spatial_id"].map(
    df.groupby("spatial_id").size()
)

print("Observations by split and temporal sequence length:")
print(
    df.groupby(["split", "num_observations"])
    .size()
    .unstack(fill_value=0)
)

split_per_location = df.groupby("spatial_id")["split"].nunique()
print("\nSpatial patches appearing in multiple splits:")
print((split_per_location > 1).sum())

print("\nS2 names:")
print(df["s2v1_name"].head(20).to_string(index=False))
print("\nS1 names:")
print(df["s1_name"].head(20).to_string(index=False))
print("\nUnique S2 names:", df["s2v1_name"].nunique())
print("Unique S1 names:", df["s1_name"].nunique())
print("\nTotal rows:", len(df))

seq = df[df["spatial_id"] == TARGET_SPATIAL_ID].sort_values("date")
print("\nTarget temporal sequence:")
print(
    seq[["patch_id", "date", "spatial_id", "s2v1_name", "s1_name", "labels"]]
    .to_string(index=False)
)

print("\nExample S2 values:")
for value in df["s2v1_name"].dropna().unique()[:20]:
    print(repr(value))

print("\nExample S1 values:")
for value in df["s1_name"].dropna().unique()[:20]:
    print(repr(value))
