
import torch
import torch.nn as nn
from torchvision.models import resnet50


class TemporalResNet50(nn.Module):

    def __init__(self, num_bands=6, num_classes=19):
        super().__init__()

        # ResNet-50 feature extractor
        self.encoder = resnet50(weights=None)

        # Replace the original 3-channel input layer
        # with a 6-channel multispectral input layer
        original_conv = self.encoder.conv1

        self.encoder.conv1 = nn.Conv2d(
            in_channels=num_bands,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        # Remove ResNet's original 1000-class classifier
        self.encoder.fc = nn.Identity()

        # Predict 19 land-cover classes
        self.classifier = nn.Linear(
            self.encoder.fc.in_features
            if isinstance(self.encoder.fc, nn.Linear)
            else 2048,
            num_classes
        )

    def forward(self, x):
        # x shape: (batch, dates, bands, height, width)
        batch_size, num_dates, num_bands, height, width = x.shape

        # Process each date using the same encoder
        x = x.reshape(
            batch_size * num_dates,
            num_bands,
            height,
            width
        )

        features = self.encoder(x)

        # Restore the temporal dimension
        features = features.reshape(
            batch_size,
            num_dates,
            -1
        )

        # Combine features across all four dates
        temporal_features = features.mean(dim=1)

        # Return raw logits, not sigmoid probabilities
        logits = self.classifier(temporal_features)

        return logits


if __name__ == "__main__":

    print("=" * 65)
    print("TESTING TEMPORAL RESNET-50")
    print("=" * 65)

    model = TemporalResNet50(
        num_bands=6,
        num_classes=19
    )

    # Simulate a batch from your real dataset
    sample_batch = torch.randn(2, 4, 6, 120, 120)

    with torch.no_grad():
        predictions = model(sample_batch)

    print("Input shape:", tuple(sample_batch.shape))
    print("Output shape:", tuple(predictions.shape))
    print("Output finite:", torch.isfinite(predictions).all().item())
    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )

    assert predictions.shape == (2, 19)
    assert torch.isfinite(predictions).all()

    print("\nSUCCESS: Model forward-pass checks passed.")
