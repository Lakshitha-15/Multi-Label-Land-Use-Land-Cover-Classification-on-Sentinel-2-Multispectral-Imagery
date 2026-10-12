
from pathlib import Path
import importlib.util
import time

from torch.utils.data import DataLoader


PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dataset_module = load_module(
    "dataset_module",
    SCRIPTS_DIR / "17_build_pytorch_dataset.py"
)

print("=" * 65)
print("SHUFFLED DATA LOADING BENCHMARK")
print("=" * 65)

dataset = dataset_module.TemporalLULCDataset(split="train")

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)

start = time.perf_counter()
count = 0

for images, targets in loader:
    count += 1

    if count == 20:
        break

    if count % 5 == 0:
        print(f"Loaded {count} batches...")


elapsed = time.perf_counter() - start

print("\nBatches loaded:", count)
print(f"Elapsed time: {elapsed:.2f} seconds")
print(f"Average loading time: {elapsed / count:.2f} seconds/batch")
print("\nBenchmark complete.")
