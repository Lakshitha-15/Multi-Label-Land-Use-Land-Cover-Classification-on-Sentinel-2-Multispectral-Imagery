
from pathlib import Path
import numpy as np
import json

PROJECT_DIR = Path(
    r"D:\Documents\Sunanda College\Fifth Semester"
    r"\ML Lab\Project\BigEarthNet-Temporal-LULC-GitReady"
    r"\BigEarthNet-Temporal-LULC"
)

DATA_DIR = PROJECT_DIR / "temporal_data"
SAMPLE_DIR = DATA_DIR / "sample_tensors"

X = np.load(SAMPLE_DIR / "sample_X.npy")

with open(
    SAMPLE_DIR / "sample_metadata.json",
    "r",
    encoding="utf-8"
) as file:
    metadata = json.load(file)

band_names = ["B02", "B03", "B04", "B08", "B11", "B12"]

print("Tensor shape:", X.shape)
print("Overall finite:", np.isfinite(X).all())
print()

print("BAND STATISTICS")
print("-" * 75)
print(
    f"{'Band':<8}{'Minimum':>12}{'Maximum':>12}"
    f"{'Mean':>14}{'Std Dev':>14}"
)

for band_index, band_name in enumerate(band_names):
    values = X[:, :, band_index, :, :]

    print(
        f"{band_name:<8}"
        f"{values.min():>12.2f}"
        f"{values.max():>12.2f}"
        f"{values.mean():>14.2f}"
        f"{values.std():>14.2f}"
    )

print("\nLABEL CHECK")
print("-" * 75)

for item in metadata[:3]:
    print("Split:", item["split"])
    print("Spatial ID:", item["spatial_id"])
    print("Dates:", item["dates"])
    print("Labels:", item["labels_by_date"][0])
    print("Label names:", item["label_names_by_date"][0])
    print("Consistent:", item["labels_consistent"])
    print()
