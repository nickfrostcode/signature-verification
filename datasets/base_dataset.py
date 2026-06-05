import os
import numpy as np
import torch
from PIL import Image

# ─────────────────────────────────────────────
# BASE DATASET
# Shared loading and preprocessing logic used
# by both PairDataset and SingleDataset.
# Not used directly — inherited by the others.
# ─────────────────────────────────────────────

# Path to processed data and stats file
PROCESSED_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'data_processed')
)
STATS_FILE = os.path.join(PROCESSED_DIR, 'stats.npz')


def load_stats():
    """
    Load the dataset mean and std computed by compute_stats.py.
    These are used to standardize every image:
        pixel = (pixel - mean) / std

    This must be called once before any image loading.
    Raises a clear error if stats.npz is missing so the user
    knows to run compute_stats.py first.
    """
    if not os.path.exists(STATS_FILE):
        raise FileNotFoundError(
            f"Stats file not found at {STATS_FILE}.\n"
            f"Run: python preprocessing/compute_stats.py"
        )
    data = np.load(STATS_FILE)
    return float(data['mean']), float(data['std'])


def load_image(img_path, mean, std):
    """
    Load a single processed PNG image and return it as a
    normalized, standardized PyTorch tensor.

    Pipeline per image:
      1. Open PNG as grayscale (already preprocessed to 155×220)
      2. Convert to float32 numpy array
      3. Scale to [0.0, 1.0] by dividing by 255
      4. Standardize: (pixel - mean) / std
      5. Add channel dimension: (H, W) → (1, H, W)
      6. Convert to PyTorch tensor

    The channel dimension (1) is required by PyTorch Conv2d layers
    which expect input shape (batch, channels, height, width).

    Input:  path to a processed PNG file
    Output: torch.Tensor shape (1, 155, 220), dtype float32
    """
    # Load as grayscale — processed images are already grayscale
    img = np.array(Image.open(img_path).convert('L')).astype(np.float32)

    # Scale to [0, 1]
    img = img / 255.0

    # Standardize using dataset-wide mean and std
    # This centers data around 0 for stable training
    img = (img - mean) / std

    # Add channel dimension: (H, W) → (1, H, W)
    img = np.expand_dims(img, axis=0)

    # Convert to PyTorch tensor
    return torch.tensor(img, dtype=torch.float32)


def get_all_subjects(processed_dir=PROCESSED_DIR):
    """
    Return a sorted list of all SIG_XXXX subject folder names.
    Used by both dataset classes to build their subject splits.
    """
    return sorted([
        d for d in os.listdir(processed_dir)
        if os.path.isdir(os.path.join(processed_dir, d))
        and d.startswith('SIG_')
    ])


def get_subject_images(processed_dir, subject, label):
    """
    Return sorted list of full paths to all images for a
    given subject and label ('genuine' or 'forged').

    Input:  processed_dir — root of data_processed/
            subject       — e.g. 'SIG_0001'
            label         — 'genuine' or 'forged'
    Output: list of absolute file paths
    """
    folder = os.path.join(processed_dir, subject, label)
    if not os.path.exists(folder):
        return []
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith('.png')
    ])


def split_subjects(subjects, train=0.8, val=0.1):
    """
    Split subjects into train, validation, and test sets.

    Split is done at the SUBJECT level — the same person never
    appears in more than one split. This is critical to prevent
    data leakage. If we split at image level, the model would
    see signatures from the same person in both train and test,
    inflating results unrealistically.

    Default split:
      80% train  → 84 subjects
      10% val    → 11 subjects
      10% test   → 10 subjects

    Input:  subjects — sorted list of subject folder names
    Output: three lists (train_subjects, val_subjects, test_subjects)
    """
    n           = len(subjects)
    train_end   = int(n * train)
    val_end     = int(n * (train + val))

    train_subjects = subjects[:train_end]
    val_subjects   = subjects[train_end:val_end]
    test_subjects  = subjects[val_end:]

    return train_subjects, val_subjects, test_subjects