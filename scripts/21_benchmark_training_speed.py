
from pathlib import Path
import importlib.util
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"

torch.manual_seed(42)
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

device = torch.device("cpu")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dataset_module = load_module(
    "dataset_module",
    SCRIPTS_DIR / "17_build_pytorch_dataset.py"
)

model_module = load_module(
    "model_module",
    SCRIPTS_DIR / "18_temporal_resnet_model.py"
)


print("=" * 65)
print("CPU TRAINING SPEED BENCHMARK")
print("=" * 65)

dataset = dataset_module.TemporalLULCDataset(split="train")

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

model = model_module.TemporalResNet50(
    num_bands=6,
    num_classes=19
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

max_batches = 5
total_start = time.perf_counter()
step_times = []

model.train()

for batch_number, (images, targets) in enumerate(loader):

    if batch_number >= max_batches:
        break

    step_start = time.perf_counter()

    images = images.to(device)
    targets = targets.to(device)

    optimizer.zero_grad()

    logits = model(images)
    loss = criterion(logits, targets)

    loss.backward()
    optimizer.step()

    elapsed = time.perf_counter() - step_start
    step_times.append(elapsed)

    print(
        f"Batch {batch_number + 1}/{max_batches} | "
        f"Loss: {loss.item():.4f} | "
        f"Step time: {elapsed:.2f} seconds"
    )

total_elapsed = time.perf_counter() - total_start

print("\n" + "-" * 65)
print(f"Batches completed: {len(step_times)}")
print(f"Total elapsed: {total_elapsed:.2f} seconds")

if step_times:
    average = sum(step_times) / len(step_times)
    print(f"Average batch step time: {average:.2f} seconds")
    print(
        "Rough training-step time for one epoch: "
        f"{average * len(dataset) / 3600:.2f} hours"
    )

print("\nBenchmark complete.")
print("Note: this estimate excludes validation and checkpoint saving.")
