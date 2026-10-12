
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class TemporalResNet50(nn.Module):
    def __init__(self, num_classes=19, pretrained=True):
        super().__init__()

        # Load pretrained ImageNet ResNet50 weights.
        weights = (
            ResNet50_Weights.DEFAULT if pretrained else None
        )
        backbone = resnet50(weights=weights)

        # Original first convolution expects RGB (3 channels).
        old_conv = backbone.conv1

        # Our input order is B02, B03, B04, B08, B11, B12.
        # Map B02 -> blue, B03 -> green, B04 -> red.
        new_conv = nn.Conv2d(
            in_channels=6,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        with torch.no_grad():
            new_conv.weight[:, 0] = old_conv.weight[:, 2]  # B02
            new_conv.weight[:, 1] = old_conv.weight[:, 1]  # B03
            new_conv.weight[:, 2] = old_conv.weight[:, 0]  # B04

            # Initialize NIR and SWIR filters from the RGB filter mean.
            mean_filter = old_conv.weight.mean(dim=1)
            for channel in range(3, 6):
                new_conv.weight[:, channel] = mean_filter

        backbone.conv1 = new_conv

        # Remove the original ImageNet classification layer.
        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-2],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.classifier = nn.Linear(
            backbone.fc.in_features, num_classes
        )

    def forward(self, x):
        # x shape: (batch, dates, bands, height, width)
        batch_size, num_dates, num_bands, height, width = x.shape

        x = x.reshape(
            batch_size * num_dates,
            num_bands,
            height,
            width
        )

        features = self.feature_extractor(x)

        # Restore date dimension and average temporal features.
        features = features.reshape(
            batch_size, num_dates, -1
        )
        temporal_features = features.mean(dim=1)

        return self.classifier(temporal_features)


if __name__ == "__main__":
    device = torch.device("cpu")

    print("Loading pretrained ResNet50...")
    model = TemporalResNet50(pretrained=True).to(device)
    model.eval()

    # Small synthetic input for an architecture check only.
    sample = torch.randn(1, 4, 6, 120, 120)

    with torch.no_grad():
        output = model(sample)

    print("Input shape:", tuple(sample.shape))
    print("Output shape:", tuple(output.shape))
    print("Output finite:", torch.isfinite(output).all().item())
    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )
    print("Pretrained temporal model test passed.")
