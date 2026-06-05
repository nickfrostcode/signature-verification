import os
from itertools import combinations
from torch.utils.data import Dataset
from datasets.base_dataset import (
    get_augmentation_transform,
    load_stats,
    load_image,
    get_subject_images,
    PROCESSED_DIR
)

# ─────────────────────────────────────────────
# PAIR DATASET — for Siamese Network
#
# Generates pairs of images with a binary label:
#   label 0 → Genuine + Genuine (same person, real match)
#   label 1 → Genuine + Forged  (mismatch, forgery attempt)
#
# Per subject with 10 genuine and 5 forged:
#   Genuine-Genuine pairs: C(10,2) = 45 pairs → label 0
#   Genuine-Forged pairs:  10 × 5 = 50 pairs  → label 1
#   Total per subject: 95 pairs
#
# For 84 training subjects:
#   84 × 95 = 7,980 training pairs
# ─────────────────────────────────────────────

import torch


class PairDataset(Dataset):

    def __init__(self, subjects, processed_dir=PROCESSED_DIR, augment=False):
        """
        Build the full list of image pairs for the given subjects.

        All pairs are generated upfront and stored as a list of
        (path_a, path_b, label) tuples. The actual images are loaded
        lazily in __getitem__ to avoid loading everything into RAM.

        Input:
            subjects      — list of subject folder names for this split
            processed_dir — root of data_processed/
            augment       — reserved for future augmentation (training only)
        """
        self.processed_dir = processed_dir
        self.augment       = augment
        self.transform     = get_augmentation_transform() if augment else None
        self.mean, self.std = load_stats()

        # Build pair list
        self.pairs  = []   # list of (path_a, path_b)
        self.labels = []   # list of int labels (0 or 1)

        for subject in subjects:
            genuine_paths = get_subject_images(processed_dir, subject, 'genuine')
            forged_paths  = get_subject_images(processed_dir, subject, 'forged')

            # Skip subject if either folder is missing or empty
            if not genuine_paths or not forged_paths:
                continue

            # ── Genuine-Genuine pairs → label 0 ──────────────────
            # combinations(genuine_paths, 2) gives all unique pairs
            # without repetition: (001,002), (001,003)... (009,010)
            # C(10,2) = 45 pairs per subject
            for path_a, path_b in combinations(genuine_paths, 2):
                self.pairs.append((path_a, path_b))
                self.labels.append(0)

            # ── Genuine-Forged pairs → label 1 ───────────────────
            # Every genuine paired with every forged
            # 10 × 5 = 50 pairs per subject
            for path_g in genuine_paths:
                for path_f in forged_paths:
                    self.pairs.append((path_g, path_f))
                    self.labels.append(1)

        print(f"PairDataset built — {len(self.pairs):,} pairs "
              f"from {len(subjects)} subjects")
        print(f"  Label 0 (genuine-genuine): "
              f"{self.labels.count(0):,}")
        print(f"  Label 1 (genuine-forged):  "
              f"{self.labels.count(1):,}")

    def __len__(self):
        """Return total number of pairs in this dataset."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Load and return one pair of images with its label.

        Called automatically by PyTorch DataLoader during training.
        Images are loaded here (lazily) rather than in __init__
        to keep memory usage low — only the current batch is in RAM.

        Output:
            img_a  — torch.Tensor (1, 155, 220)
            img_b  — torch.Tensor (1, 155, 220)
            label  — torch.Tensor scalar (0 or 1), float32
        """
        path_a, path_b = self.pairs[idx]
        label          = self.labels[idx]

        img_a = load_image(path_a, self.mean, self.std, transform=self.transform)
        img_b = load_image(path_b, self.mean, self.std, transform=self.transform)

        # Label as float32 — required by BCELoss in PyTorch
        label = torch.tensor(label, dtype=torch.float32)

        return img_a, img_b, label