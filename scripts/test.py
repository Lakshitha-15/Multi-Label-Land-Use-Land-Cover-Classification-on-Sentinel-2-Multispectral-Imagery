
from pathlib import Path

root = Path(
    r"D:\Documents\Sunanda College\Fifth Semester\ML Lab\Project"
)

filename = "sample_image_lookup.csv"

print("Searching for", filename)

found = False

for path in root.rglob(filename):
    print("FOUND:", path)
    found = True

if not found:
    print("CSV not found in the project directory.")
    print("We may need to regenerate it.")
