"""Inspect one known four-observation spatial location chronologically."""
import pandas as pd

METADATA_PATH = "data/metadata.parquet"
TARGET_SPATIAL_ID = "T34TCR_45_52"

df = pd.read_parquet(METADATA_PATH)
df["date"] = pd.to_datetime(
    df["patch_id"].str.extract(r"MSIL2A_(\d{8})")[0],
    format="%Y%m%d",
)
df["tile"] = df["patch_id"].str.extract(r"(T\d{2}[A-Z]{3})")
df["spatial_id"] = df["patch_id"].str.extract(
    r"(T\d{2}[A-Z]{3}_\d+_\d+)$"
)

seq = df[df["spatial_id"] == TARGET_SPATIAL_ID].sort_values("date")

print("\nNumber of observations:", len(seq))
print(
    seq[
        [
            "patch_id",
            "date",
            "tile",
            "spatial_id",
            "s2v1_name",
            "s1_name",
            "labels",
        ]
    ].to_string(index=False)
)
