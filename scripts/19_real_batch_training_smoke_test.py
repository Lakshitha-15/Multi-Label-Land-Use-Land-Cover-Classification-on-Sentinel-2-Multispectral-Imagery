
from pathlib import Path
import importlib.util
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"


# ------------------------------------------------------------
# LOAD EXISTING SCRIPTS
# Their filenames start with numbers, so use importlib.
# ------------------------------------------------------------

def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(
        module_name, file_path
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


dataset_module = load_module(
    "temporal_dataset_module",
    SCRIPTS_DIR / "17_build_pytorch_dataset.py"
)

model_module = load_module(
    "temporal_model_module",
    SCRIPTS_DIR / "18_temporal_resnet_model.py"
)


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

BATCH_SIZE = 1
LEARNING_RATE = 1e-4

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(42)

print("=" * 65)
print("REAL-DATA TRAINING SMOKE TEST")
print("=" * 65)
print("Device:", device)


# ------------------------------------------------------------
# LOAD ONE REAL TRAINING BATCH
# ------------------------------------------------------------

print("\nLoading training dataset...")

train_dataset = dataset_module.TemporalLULCDataset(
    split="train"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

images, targets = next(iter(train_loader))

images = images.to(device)
targets = targets.to(device)

print("Images:", tuple(images.shape))
print("Targets:", tuple(targets.shape))
print("Positive labels:", targets.sum(dim=1).tolist())


# ------------------------------------------------------------
# INITIALIZE MODEL, LOSS, AND OPTIMIZER
# ------------------------------------------------------------

model = model_module.TemporalResNet50(
    num_bands=6,
    num_classes=19
).to(device)

# Load class frequencies calculated from temporal sequences
class_df = pd.read_csv(
    PROJECT_DIR / "temporal_data" / "class_distribution.csv"
)

class_df = class_df.sort_values("class_id")

train_counts = torch.tensor(
    class_df["train_count"].to_numpy(),
    dtype=torch.float32,
    device=device
)

total_train = 14812

pos_weight = (
    (total_train - train_counts) / train_counts
).clamp(min=1.0, max=20.0)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

print("Class weights:", pos_weight.cpu().tolist())

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ------------------------------------------------------------
# ONE TRAINING STEP
# ------------------------------------------------------------

model.train()

parameters_before = (
    model.classifier.weight.detach().clone()
)

optimizer.zero_grad()

logits = model(images)
loss = criterion(logits, targets)

print("\nLoss:", loss.item())
print("Logits finite:", torch.isfinite(logits).all().item())
print("Loss finite:", torch.isfinite(loss).item())

assert torch.isfinite(loss), "Loss is not finite!"

loss.backward()

# Verify that gradients exist and are finite
gradient = model.classifier.weight.grad

assert gradient is not None, "Missing classifier gradients!"
assert torch.isfinite(gradient).all(), "Non-finite gradients!"

optimizer.step()

parameters_after = model.classifier.weight.detach()

parameters_updated = not torch.equal(
    parameters_before,
    parameters_after
)

print("Gradients finite:", torch.isfinite(gradient).all().item())
print("Parameters updated:", parameters_updated)

assert parameters_updated, "Model parameters did not update!"

print("\nSUCCESS: Real-data training step passed.")
