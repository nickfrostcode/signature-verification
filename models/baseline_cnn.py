import torch
import torch.nn as nn

# ─────────────────────────────────────────────
# BASELINE CNN — Lightweight Binary Classifier
#
# Reduced from 16M → ~400K parameters.
# Smaller capacity forces generalization
# instead of memorization on small datasets.
# ─────────────────────────────────────────────


class BaselineCNN(nn.Module):
    """
    Lightweight 3-block CNN for binary signature classification.

    Deliberately small to prevent overfitting on 1,260 images.
    Uses GlobalAveragePooling instead of a large FC layer
    to reduce parameters while preserving spatial reasoning.

    Spatial flow:
        Input:   (1, 155, 220)
        Block 1: (16,  77, 110)
        Block 2: (32,  38,  55)
        Block 3: (64,  19,  27)
        GAP:     (64,   1,   1)  ← Global Average Pool
        Flatten: 64
        FC:      1 output logit
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1 — low level edges and stroke tips
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.1),

            # Block 2 — mid level curves and connections
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.1),

            # Block 3 — high level signature structure
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.2),
        )

        # Global Average Pooling — replaces large FC layer
        # Averages each feature map to a single value
        # Reduces 64×19×27=32,832 → 64 with no information bottleneck
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(32, 1),
            # No sigmoid — BCEWithLogitsLoss handles it
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


if __name__ == '__main__':
    model = BaselineCNN()
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"BaselineCNN — parameters: {total:,}")
    dummy = torch.randn(8, 1, 155, 220)
    out   = model(dummy)
    print(f"Input: {dummy.shape} → Output: {out.shape}")
    print("✅ Forward pass successful")