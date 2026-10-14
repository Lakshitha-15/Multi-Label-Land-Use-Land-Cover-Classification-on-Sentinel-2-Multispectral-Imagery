
from pathlib import Path
import importlib.util

import torch


PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"

device = torch.device("cpu")


def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(
        module_name, file_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dataset_module = load_module(
    "smoke_test_dataset",
    SCRIPTS_DIR / "17_build_pytorch_dataset.py"
)

model_module = load_module(
    "smoke_test_model",
    SCRIPTS_DIR / "23_test_pretrained_temporal_model.py"
)

print("Loading one training sample...")
dataset = dataset_module.TemporalLULCDataset(split="train")
images, targets = dataset[0]

print("Image shape:", tuple(images.shape))
print("Label shape:", tuple(targets.shape))

assert tuple(images.shape) == (4, 6, 120, 120)
assert tuple(targets.shape) == (19,)

print("\nLoading frozen ResNet50...")
model = model_module.TemporalResNet50(
    num_classes=19,
    pretrained=True
).to(device)

model.eval()

for parameter in model.parameters():
    parameter.requires_grad = False

images = images.unsqueeze(0).to(device)

with torch.no_grad():
    batch_size, num_dates, num_bands, height, width = images.shape

    flattened = images.reshape(
        batch_size * num_dates,
        num_bands,
        height,
        width
    )

    features = model.feature_extractor(flattened)
    features = features.reshape(batch_size, num_dates, -1)

print("\nExtracted feature shape:", tuple(features.shape))
print("All features finite:", torch.isfinite(features).all().item())

assert tuple(features.shape) == (1, 4, 2048)
assert torch.isfinite(features).all().item()

print("\nSMOKE TEST PASSED")
