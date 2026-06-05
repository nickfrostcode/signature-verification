import os
from torch.utils.data import Dataset
from datasets.base_dataset import (
    load_stats,
    load_image,
    get_subject_images,
    PROCESSED_DIR
)

# ─────────────────────────────────────────────
# SINGLE DATASET — for Baseline CNN
#
# Each item is one image with a direct label:
#   label 0 → genuine image
#   label 1 → forged image
#
# Per subject:
#   10 genuine images → label 0
#    5 forged images  → label 1
#   Total: 15 images per subject
#
# For 84 training subjects:
#   84 × 15 = 1,260 training images
#
# Note on class imbalance:
#   Genuine images are 2× more than forged (10 vs 5).
#   This is handled during training using class weights.
# ─────────────────────────────────────────────

import torch


class SingleDataset(Dataset):

    def __init__(self, subjects, processed_dir=PROCESSED_DIR):
        """
        Build the full list of individual images for the given subjects.

        Each entry is a (image_path, label) tuple.
        Images are loaded lazily in __getitem__.

        Input:
            subjects      — list of subject folder names for this split
            processed_dir — root of data_processed/
        """
        self.processed_dir = processed_dir
        self.mean, self.std = load_stats()

        self.image_paths = []   # list of file paths
        self.labels      = []   # list of int labels (0 or 1)

        for subject in subjects:
            genuine_paths = get_subject_images(processed_dir, subject, 'genuine')
            forged_paths  = get_subject_images(processed_dir, subject, 'forged')

            # Genuine images → label 0
            for path in genuine_paths:
                self.image_paths.append(path)
                self.labels.append(0)

            # Forged images → label 1
            for path in forged_paths:
                self.image_paths.append(path)
                self.labels.append(1)

        print(f"SingleDataset built — {len(self.image_paths):,} images "
              f"from {len(subjects)} subjects")
        print(f"  Label 0 (genuine): {self.labels.count(0):,}")
        print(f"  Label 1 (forged):  {self.labels.count(1):,}")

    def __len__(self):
        """Return total number of images in this dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Load and return one image with its label.

        Output:
            img   — torch.Tensor (1, 155, 220)
            label — torch.Tensor scalar (0 or 1), float32
        """
        path  = self.image_paths[idx]
        label = self.labels[idx]

        img   = load_image(path, self.mean, self.std)

        # Label as float32 — required by BCELoss in PyTorch
        label = torch.tensor(label, dtype=torch.float32)

        return img, label