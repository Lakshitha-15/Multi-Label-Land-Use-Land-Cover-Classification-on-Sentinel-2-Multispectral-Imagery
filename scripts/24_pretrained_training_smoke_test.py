
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
DATA_DIR = PROJECT_DIR / "temporal_data"


# ------------------------------------------------------------
# LOAD EXISTING SCRIPTS
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

pretrained_module = load_module(
    "pretrained_temporal_model_module",
    SCRIPTS_DIR / "23_test_pretrained_temporal_model.py"
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
print("PRETRAINED REAL-DATA TRAINING SMOKE TEST")
print("=" * 65)
print("Device:", device)


# ------------------------------------------------------------
# LOAD ONE REAL TRAINING BATCH
# ------------------------------------------------------------

print("\nLoading real training data...")

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

print("Training sequences:", len(train_dataset))
print("Images:", tuple(images.shape))
print("Targets:", tuple(targets.shape))
print("Positive labels:", targets.sum(dim=1).tolist())

assert images.shape[1:] == (4, 6, 120, 120)
assert targets.shape[1] == 19
assert torch.isfinite(images).all()
assert torch.isfinite(targets).all()


# ------------------------------------------------------------
# INITIALIZE PRETRAINED MODEL
# ------------------------------------------------------------

print("\nLoading pretrained temporal ResNet50...")

model = pretrained_module.TemporalResNet50(
    num_classes=19,
    pretrained=True
).to(device)


# ------------------------------------------------------------
# FREEZE FEATURE EXTRACTOR
# Only the final classifier will be trained in this test.
# ------------------------------------------------------------

for parameter in model.feature_extractor.parameters():
    parameter.requires_grad = False

# Keep frozen BatchNorm layers in evaluation mode.
model.train()
model.feature_extractor.eval()

trainable_parameters = [
    parameter
    for parameter in model.parameters()
    if parameter.requires_grad
]

print(
    "Trainable parameters:",
    sum(p.numel() for p in trainable_parameters)
)


# ------------------------------------------------------------
# LOAD CLASS WEIGHTS
# ------------------------------------------------------------

class_df = pd.read_csv(
    DATA_DIR / "class_distribution.csv"
).sort_values("class_id")

assert class_df["class_id"].tolist() == list(range(19))

train_counts = torch.tensor(
    class_df["train_count"].to_numpy(),
    dtype=torch.float32,
    device=device
)

total_train = len(train_dataset)

pos_weight = (
    (total_train - train_counts) / train_counts
).clamp(min=1.0, max=20.0)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

print("Class weights:", pos_weight.cpu().tolist())

optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=LEARNING_RATE
)


# ------------------------------------------------------------
# ONE REAL-DATA TRAINING STEP
# ------------------------------------------------------------

classifier_before = model.classifier.weight.detach().clone()

optimizer.zero_grad()

logits = model(images)
loss = criterion(logits, targets)

print("\nLoss:", loss.item())
print("Logits finite:", torch.isfinite(logits).all().item())
print("Loss finite:", torch.isfinite(loss).item())

assert torch.isfinite(logits).all()
assert torch.isfinite(loss)

loss.backward()

gradient = model.classifier.weight.grad

assert gradient is not None, "Missing classifier gradients!"
assert torch.isfinite(gradient).all(), "Non-finite gradients!"

optimizer.step()

classifier_after = model.classifier.weight.detach()

classifier_updated = not torch.equal(
    classifier_before,
    classifier_after
)

print("Gradients finite:", torch.isfinite(gradient).all().item())
print("Classifier updated:", classifier_updated)

assert classifier_updated, "Classifier parameters did not update!"

# Confirm the frozen feature extractor received no gradients.
backbone_has_gradients = any(
    parameter.grad is not None
    for parameter in model.feature_extractor.parameters()
)

print("Frozen backbone has gradients:", backbone_has_gradients)

assert not backbone_has_gradients

print("\nSUCCESS: Pretrained real-data training step passed.")
