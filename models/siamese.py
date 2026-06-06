import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseEncoder(nn.Module):
    """
    Medium-sized shared CNN encoder.

    Spatial flow:
        Input:   (1, 155, 220)
        Block 1: (32,  77, 110)
        Block 2: (64,  38,  55)
        Block 3: (128, 19,  27)
        GAP:     (128,  1,   1)
        Flatten: 128 → 64-dim normalized embedding
    """

    def __init__(self):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.15),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.2),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
        )

    def forward(self, x):
        x   = self.conv_blocks(x)
        x   = self.gap(x)
        emb = self.embedding(x)
        # L2 normalize — keeps embeddings on unit sphere
        return F.normalize(emb, p=2, dim=1)


class SiameseNetwork(nn.Module):
    """
    Siamese Network with L1 distance + BCE classifier.

    Uses BCE on the L1 distance vector instead of contrastive loss.
    More stable on small datasets than metric learning approaches.

    Architecture:
        img_a → Encoder → emb_a ──┐
                                    ├→ L1 distance → classifier → logit
        img_b → Encoder → emb_b ──┘
        (shared weights)
    """

    def __init__(self):
        super().__init__()
        self.encoder = SiameseEncoder()

        # Small classifier on top of L1 distance vector (64-dim)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(32, 1),
            # No sigmoid — BCEWithLogitsLoss handles it
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, img_a, img_b):
        emb_a    = self.encoder(img_a)
        emb_b    = self.encoder(img_b)
        # L1 distance — absolute difference per embedding dimension
        distance = torch.abs(emb_a - emb_b)
        return self.classifier(distance)


if __name__ == '__main__':
    model  = SiameseNetwork()
    total  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SiameseNetwork — parameters: {total:,}")
    img_a  = torch.randn(8, 1, 155, 220)
    img_b  = torch.randn(8, 1, 155, 220)
    out    = model(img_a, img_b)
    emb    = model.encode(img_a)
    print(f"Input: {img_a.shape} → Output: {out.shape}")
    print(f"Embedding: {emb.shape}")
    print("✅ Forward pass successful")