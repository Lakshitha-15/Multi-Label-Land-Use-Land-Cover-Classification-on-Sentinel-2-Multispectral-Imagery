
from pathlib import Path
import pandas as pd

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

PROJECT_DIR = Path(
    r"D:\Documents\Sunanda College\Fifth Semester"
    r"\ML Lab\Project"
)

INDEX_DIR = PROJECT_DIR / "temporal_data"

# ------------------------------------------------------------
# CHECK FOUR-DATE PATCHES IN EACH SPLIT
# ------------------------------------------------------------

def inspect_four_date_patches(split_name):

    index_file = INDEX_DIR / f"{split_name}_index.csv"
    df = pd.read_csv(index_file)

    # Keep only spatial patches with exactly four observations.
    four_date_df = df[df["num_observations"] == 4].copy()

    # Confirm each selected patch has four distinct dates.
    date_counts = (
        four_date_df.groupby("spatial_id")["date"]
        .nunique()
    )

    valid_locations = date_counts[date_counts == 4].index
    four_date_df = four_date_df[
        four_date_df["spatial_id"].isin(valid_locations)
    ]

    print(f"\n{'=' * 55}")
    print(f"{split_name.upper()} SPLIT")
    print("=" * 55)

    print("Four-observation locations:",
          df.loc[df["num_observations"] == 4, "spatial_id"].nunique())

    print("Locations with four distinct dates:",
          len(valid_locations))

    print("\nExample temporal sequences:")

    for spatial_id, group in (
        four_date_df.groupby("spatial_id", sort=False)
    ):
        group = group.sort_values("date")

        print(f"\nSpatial patch: {spatial_id}")

        for _, row in group.iterrows():
            print(
                f"  Date: {row['date']} | "
                f"Image: {row['image_id']}"
            )

        # Print only a few examples.
        if spatial_id == valid_locations[2]:
            break

    return four_date_df


if __name__ == "__main__":

    train_four = inspect_four_date_patches("train")
    val_four = inspect_four_date_patches("validation")
    test_four = inspect_four_date_patches("test")

    print("\n" + "=" * 55)
    print("FOUR-DATE PATCH SUMMARY")
    print("=" * 55)

    print("Train:", train_four["spatial_id"].nunique())
    print("Validation:", val_four["spatial_id"].nunique())
    print("Test:", test_four["spatial_id"].nunique())
