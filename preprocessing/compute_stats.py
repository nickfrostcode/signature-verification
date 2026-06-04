import os
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
PROCESSED_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'data_processed')
)
STATS_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'data_processed', 'stats.npz')
)


def compute_stats(processed_dir):
    """
    Make one pass over all processed images and compute
    the global mean and standard deviation of pixel values.

    These two numbers are used during training and inference
    to standardize every image:
        pixel = (pixel - mean) / std

    Why compute from scratch instead of using ImageNet stats:
      - ImageNet mean/std are for RGB natural photos
      - Our images are grayscale binary signature scans
      - Using wrong stats would shift pixel distributions incorrectly

    Only processes training subjects (first 80%) to avoid
    data leakage from validation/test sets into the stats.
    """

    subjects = sorted([
        d for d in os.listdir(processed_dir)
        if os.path.isdir(os.path.join(processed_dir, d))
        and d.startswith('SIG_')
    ])

    # Use only training subjects (first 80%) for stats computation
    # This prevents validation/test data from influencing normalization
    train_end  = int(0.8 * len(subjects))
    train_subjects = subjects[:train_end]

    print(f"Total subjects:    {len(subjects)}")
    print(f"Training subjects: {len(train_subjects)} (used for stats)")
    print(f"Held-out subjects: {len(subjects) - len(train_subjects)} (val + test)\n")

    pixel_sum    = 0.0   # running sum of all pixel values
    pixel_sq_sum = 0.0   # running sum of squared pixel values
    pixel_count  = 0     # total number of pixels seen

    for subject in train_subjects:
        for label in ['genuine', 'forged']:
            folder = os.path.join(processed_dir, subject, label)
            if not os.path.exists(folder):
                continue

            for fname in os.listdir(folder):
                if not fname.endswith('.png'):
                    continue

                fpath = os.path.join(folder, fname)
                try:
                    # Load as grayscale float
                    img        = np.array(Image.open(fpath)).astype(np.float32) / 255.0
                    pixel_sum    += img.sum()
                    pixel_sq_sum += (img ** 2).sum()
                    pixel_count  += img.size

                except Exception as e:
                    print(f"  ⚠️  Could not read {fpath}: {e}")

    if pixel_count == 0:
        print("❌ No images found. Check your data_processed/ folder.")
        return

    # Mean = sum / count
    mean = pixel_sum / pixel_count

    # Std = sqrt(E[x²] - E[x]²)
    # This is the numerically stable one-pass formula
    std  = np.sqrt((pixel_sq_sum / pixel_count) - (mean ** 2))

    print(f"{'=' * 40}")
    print(f"DATASET STATISTICS (training set only)")
    print(f"{'=' * 40}")
    print(f"  Pixels scanned: {pixel_count:,}")
    print(f"  Mean:           {mean:.6f}")
    print(f"  Std:            {std:.6f}")
    print(f"{'=' * 40}\n")

    # Save to file — these values are loaded by the dataset class during training
    np.savez(STATS_FILE, mean=mean, std=std)
    print(f"  ✅ Stats saved to: {STATS_FILE}")
    print(f"     Load with: data = np.load('stats.npz')")
    print(f"                mean, std = data['mean'], data['std']")


if __name__ == '__main__':
    print("=" * 40)
    print("COMPUTING DATASET MEAN AND STD")
    print("=" * 40 + "\n")
    compute_stats(PROCESSED_DIR)