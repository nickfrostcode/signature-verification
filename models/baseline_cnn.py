import torch
import torch.nn as nn

# ─────────────────────────────────────────────
# BASELINE CNN — Binary Classifier
#
# The simplest possible approach:
# Feed one signature image → get genuine/forged prediction.
#
# Architecture:
#   3 convolutional blocks (feature extraction)
#   2 fully connected layers (classification)
#
# This model has no concept of "whose signature is this" —
# it just learns visual patterns that distinguish genuine
# strokes from forged ones globally across all subjects.
#
# Used as the performance baseline.
# The Siamese network should outperform this.
# ─────────────────────────────────────────────


class ConvBlock(nn.Module):
    """
    A single convolutional block used repeatedly in the network.

    Structure:
        Conv2d → BatchNorm → ReLU → MaxPool

    Why BatchNorm:
        Normalizes activations between layers.
        Stabilizes training, allows higher learning rates,
        reduces sensitivity to weight initialization.

    Why MaxPool:
        Reduces spatial dimensions by 2x after each block.
        Forces the network to learn position-invariant features.
        A stroke slightly left or right should still be detected.

    Input:  (batch, in_channels,  H,   W)
    Output: (batch, out_channels, H/2, W/2)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            # Conv2d learns spatial filters (edge detectors, stroke detectors)
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            # BatchNorm normalizes output of conv across the batch
            nn.BatchNorm2d(out_channels),
            # ReLU introduces non-linearity — without this the network
            # is just a series of linear transformations (useless for complex patterns)
            nn.ReLU(inplace=True),
            # MaxPool halves spatial size, keeps strongest activations
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)


class BaselineCNN(nn.Module):
    """
    3-block CNN for binary signature classification.

    Input:  (batch, 1, 155, 220) — grayscale signature tensor
    Output: (batch, 1)           — score between 0 and 1
                                   0 = genuine, 1 = forged

    Spatial flow through conv blocks:
        Input:   (1, 155, 220)
        Block 1: (32,  77, 110)   ← 32 feature maps, halved spatial
        Block 2: (64,  38,  55)   ← 64 feature maps, halved again
        Block 3: (128, 19,  27)   ← 128 feature maps, halved again
        Flatten: 128 × 19 × 27 = 65,664 features
        FC1:     256 features
        FC2:     1 output (sigmoid score)
    """

    def __init__(self):
        super().__init__()

        # ── Feature Extractor ─────────────────────────────────
        # Three conv blocks progressively extract higher-level features:
        #   Block 1: low-level — edges, stroke tips, endpoints
        #   Block 2: mid-level — stroke curves, loops, connections
        #   Block 3: high-level — signature regions, overall structure
        self.features = nn.Sequential(
            ConvBlock(1,   32),    # 1 input channel (grayscale)
            ConvBlock(32,  64),
            ConvBlock(64, 128),
        )

        # ── Classifier ────────────────────────────────────────
        # Takes flattened feature vector → binary prediction
        self.classifier = nn.Sequential(
            # Flatten (128, 19, 27) → (65664,)
            nn.Flatten(),

            # FC layer 1: compress features
            nn.Linear(128 * 19 * 27, 256),
            nn.ReLU(inplace=True),

            # Dropout: randomly zeros 50% of neurons during training
            # Forces the network to not rely on any single feature
            # Reduces overfitting significantly on small datasets
            nn.Dropout(p=0.5),

            # FC layer 2: final binary output
            nn.Linear(256, 1),
        )

    def forward(self, x):
        """
        Forward pass — runs input through feature extractor then classifier.

        Input:  x — tensor (batch, 1, 155, 220)
        Output: tensor (batch, 1) — forgery probability per image
        """
        features = self.features(x)
        return self.classifier(features)


# ─────────────────────────────────────────────
# QUICK ARCHITECTURE VERIFICATION
# Run directly: python -m models.baseline_cnn
# ─────────────────────────────────────────────
if __name__ == '__main__':
    import torch

    model = BaselineCNN()

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"BaselineCNN")
    print(f"  Total trainable parameters: {total_params:,}")

    # Test forward pass with dummy batch
    dummy = torch.randn(8, 1, 155, 220)   # batch of 8 images
    out   = model(dummy)
    print(f"  Input shape:  {dummy.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
    print(f"  ✅ Forward pass successful")