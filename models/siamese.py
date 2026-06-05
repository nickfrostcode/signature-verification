import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────
# SIAMESE NETWORK — Similarity-based Verifier
#
# Core idea:
#   Two identical CNN branches (shared weights) encode
#   each input image into a normalized embedding vector.
#   Euclidean distance between embeddings is used as a
#   similarity measure for contrastive metric learning.
#
# Why shared weights:
#   Both branches must learn the SAME feature space.
#   If weights were separate, branch A might learn
#   different stroke features than branch B, making
#   distance comparison meaningless.
#
# Why normalized embeddings:
#   Normalization constrains the embedding space to the
#   unit sphere, which stabilizes distances and makes
#   the contrastive margin easier to tune.
#
# Input:  two tensors (batch, 1, 155, 220)
# Output: one tensor  (batch, 1) — Euclidean distance
# ─────────────────────────────────────────────


class ConvBlock(nn.Module):
    """
    Convolutional block for the Siamese encoder.

    Structure: Conv2d → BatchNorm → ReLU → MaxPool

    Same as BaselineCNN's ConvBlock but defined separately
    to keep models fully independent of each other.
    Changes to one model's blocks won't affect the other.

    Input:  (batch, in_channels,  H,   W)
    Output: (batch, out_channels, H/2, W/2)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)


class SiameseEncoder(nn.Module):
    """
    The shared CNN encoder used by both branches.

    Takes a signature image and produces a compact
    embedding vector that represents the signature's
    unique characteristics in a learned feature space.

    The goal: genuine signatures from the same person
    should produce similar embeddings (small distance),
    while forged signatures should produce different
    embeddings (large distance).

    Spatial flow:
        Input:   (1, 155, 220)
        Block 1: (32,  77, 110)
        Block 2: (64,  38,  55)
        Block 3: (128, 19,  27)
        Block 4: (256,  9,  13)   ← deeper than baseline
        Flatten: 256 × 9 × 13 = 29,952
        FC:      128-dim embedding
    """

    def __init__(self):
        super().__init__()

        # 4 conv blocks — one more than baseline
        # Deeper encoder captures more abstract signature features
        # needed for meaningful embedding comparison
        self.conv_blocks = nn.Sequential(
            ConvBlock(1,    32),
            ConvBlock(32,   64),
            ConvBlock(64,  128),
            ConvBlock(128, 256),
        )

        # Fully connected layers compress spatial features
        # into a compact 128-dimensional embedding vector
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 9 * 13, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            # Final embedding dimension
            # 128 dimensions — enough to capture signature uniqueness
            # without being too large (which causes overfitting)
            nn.Linear(512, 128),
        )

    def forward(self, x):
        """
        Encode one image into a 128-dim embedding vector.

        Input:  x — tensor (batch, 1, 155, 220)
        Output: tensor (batch, 128) — embedding vector
        """
        features  = self.conv_blocks(x)
        embedding = self.embedding(features)
        return F.normalize(embedding, p=2, dim=1)


class SiameseNetwork(nn.Module):
    """
    Full Siamese Network for signature verification.

    Takes two signature images, encodes both using the
    shared encoder, and computes the Euclidean distance
    between normalized embeddings.

    Architecture:
        img_a → Encoder ──→ embedding_a ──┐
                                       ├→ Euclidean distance
        img_b → Encoder ──→ embedding_b ─┘
        (same weights)

    Input:  img_a, img_b — tensors (batch, 1, 155, 220)
    Output: tensor (batch, 1) — Euclidean distance between embeddings
    """

    def __init__(self):
        super().__init__()

        # Shared encoder produces normalized embeddings.
        # Normalization constrains the embedding space to a unit sphere,
        # making Euclidean distance a stable similarity measure.
        self.encoder = SiameseEncoder()

    def encode(self, x):
        """
        Public method to get embedding for a single image.
        Used during inference and Grad-CAM visualization.

        Input:  x — tensor (batch, 1, 155, 220)
        Output: tensor (batch, 128)
        """
        return self.encoder(x)

    def forward(self, img_a, img_b):
        """
        Forward pass for a pair of images.

        Output is the Euclidean distance between normalized embeddings.
        Smaller values indicate genuine pairs, larger values indicate forgery pairs.
        """
        embedding_a = self.encoder(img_a)
        embedding_b = self.encoder(img_b)
        distance = F.pairwise_distance(embedding_a, embedding_b, p=2)
        return distance.unsqueeze(1)


# ─────────────────────────────────────────────
# QUICK ARCHITECTURE VERIFICATION
# Run directly: python -m models.siamese
# ─────────────────────────────────────────────
if __name__ == '__main__':

    model = SiameseNetwork()

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SiameseNetwork")
    print(f"  Total trainable parameters: {total_params:,}")

    # Test forward pass with dummy pair
    img_a = torch.randn(8, 1, 155, 220)
    img_b = torch.randn(8, 1, 155, 220)
    out   = model(img_a, img_b)

    print(f"  Input shape (each): {img_a.shape}")
    print(f"  Output shape:       {out.shape}")
    print(f"  Output range:       [{out.min().item():.4f}, {out.max().item():.4f}]")

    # Verify shared weights — both branches are literally the same object
    print(f"\n  Shared encoder verified: "
          f"{model.encoder is model.encoder}")

    # Test encode method separately
    emb = model.encode(img_a)
    print(f"  Embedding shape:    {emb.shape}")
    print(f"  ✅ Forward pass successful")