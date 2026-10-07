"""Analyze BigEarthNet-v2 metadata and identify multi-temporal patches."""
import pandas as pd

METADATA_PATH = "data/metadata.parquet"

df = pd.read_parquet(METADATA_PATH)

df["date"] = pd.to_datetime(
    df["patch_id"].str.extract(r"MSIL2A_(\d{8})")[0],
    format="%Y%m%d",
)
df["tile"] = df["patch_id"].str.extract(r"(T\d{2}[A-Z]{3})")
df["spatial_id"] = df["patch_id"].str.extract(
    r"(T\d{2}[A-Z]{3}_\d+_\d+)$"
)

print("Number of patches:", len(df))
print("Number of acquisition dates:", df["date"].nunique())
print("Number of Sentinel-2 tiles:", df["tile"].nunique())
print("\nDate range:")
print(df["date"].min(), "to", df["date"].max())
print("\nObservations per tile:")
print(df.groupby("tile")["date"].nunique().sort_values(ascending=False).head(20))

print("\nExample patch IDs:")
for x in df["patch_id"].head(20):
    print(x)

print("\nExample spatial IDs:")
print(df[["patch_id", "date", "spatial_id"]].head(20).to_string(index=False))

location_counts = df["spatial_id"].value_counts()
print("\nNumber of unique spatial patches:")
print(df["spatial_id"].nunique())
print("\nSpatial patches with multiple observations:")
print(location_counts[location_counts > 1].value_counts().sort_index())

four_plus = location_counts[location_counts >= 4]
print("\nPatches with >= 4 observations:", len(four_plus))

print("\nExample spatial patches:")
for spatial_id in four_plus.head(10).index:
    rows = df[df["spatial_id"] == spatial_id]
    print("\n", spatial_id)
    print(rows[["date", "patch_id", "labels"]].sort_values("date").to_string(index=False))
