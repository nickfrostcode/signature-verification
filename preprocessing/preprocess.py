import os
import cv2
import numpy as np
from PIL import Image

"""Preprocessing utilities for signature image normalization.

This module loads raw signature scans, converts them to a clean
binary format, crops and pads each signature, and saves processed
images ready for model training or inference.
"""

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
DATA_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processed'))

# Target size (height x width) — standard in signature verification research
TARGET_H = 155
TARGET_W = 220

# Padding margin to add around the cropped signature content (in pixels)
CROP_PADDING = 10


# ─────────────────────────────────────────────
# STEP 1 — GRAYSCALE CONVERSION
# ─────────────────────────────────────────────
def to_grayscale(img_rgb):
    """
    Convert an RGB image to grayscale.
    Signatures are ink on paper — color carries zero useful information.
    Grayscale reduces the image from 3 channels to 1, cutting
    memory and computation without losing anything meaningful.

    Input:  numpy array (H, W, 3) — RGB
    Output: numpy array (H, W)    — grayscale, values 0-255
    """
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


# ─────────────────────────────────────────────
# STEP 2 — OTSU BINARIZATION
# ─────────────────────────────────────────────
def binarize(gray_img):
    """
    Apply Otsu's thresholding to convert grayscale to pure black/white.

    Why Otsu specifically:
      - It automatically finds the optimal threshold per image
      - No hardcoded value needed — works on any lighting condition
      - Two signatures of the same person scanned on different scanners
        will produce very different gray values, but after Otsu they
        both become the same clean black stroke on white paper

    Otsu works by analyzing the image histogram and finding the threshold
    that minimizes within-class variance between foreground and background.

    Input:  numpy array (H, W) — grayscale
    Output: numpy array (H, W) — binary, values are either 0 or 255
    """
    # cv2.THRESH_BINARY + cv2.THRESH_OTSU lets OpenCV compute
    # the threshold automatically from the image histogram
    _, binary = cv2.threshold(
        gray_img, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


# ─────────────────────────────────────────────
# STEP 3 — INVERT CHECK
# ─────────────────────────────────────────────
def ensure_white_background(binary_img):
    """
    Ensure the image has a WHITE background and BLACK strokes.
    This is the standard convention for signature images.

    Some scanners or cameras produce inverted output:
      - Black background, white strokes (inverted)
    We need to detect and correct this.

    Detection method:
      - Count white pixels vs black pixels
      - If black pixels outnumber white pixels, the image is inverted
      - Invert it so background becomes white

    Input:  numpy array (H, W) — binary image (0 or 255)
    Output: numpy array (H, W) — binary image, white bg guaranteed
    """
    white_pixels = np.sum(binary_img == 255)
    black_pixels = np.sum(binary_img == 0)

    # In a normal signature image, background (white) should dominate
    # If black dominates, the image is inverted
    if black_pixels > white_pixels:
        return cv2.bitwise_not(binary_img)

    return binary_img


# ─────────────────────────────────────────────
# STEP 4 — CROP TO CONTENT
# ─────────────────────────────────────────────
def crop_to_content(binary_img, padding=CROP_PADDING):
    """
    Find the bounding box of all black pixels (the actual signature strokes)
    and crop to that region plus a small padding margin.

    Why this matters:
      - Raw scans often have large empty white borders around the signature
      - These borders add no information but waste model capacity
      - Cropping focuses the model entirely on the ink strokes
      - The padding prevents the signature from touching the edge,
        which can confuse convolutional layers at borders

    If no black pixels are found (completely blank image), the original
    image is returned unchanged to avoid crashes.

    Input:  numpy array (H, W) — binary image
    Output: numpy array (H, W) — cropped binary image
    """
    # Find all pixel coordinates where value is 0 (black = ink stroke)
    black_pixels = np.where(binary_img == 0)

    # Guard: if no black pixels found, return original
    if len(black_pixels[0]) == 0:
        return binary_img

    # Get bounding box of all black pixels
    y_min = int(np.min(black_pixels[0]))
    y_max = int(np.max(black_pixels[0]))
    x_min = int(np.min(black_pixels[1]))
    x_max = int(np.max(black_pixels[1]))

    # Add padding, clamped to image boundaries so we don't go out of bounds
    h, w  = binary_img.shape
    y_min = max(0, y_min - padding)
    y_max = min(h, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(w, x_max + padding)

    return binary_img[y_min:y_max, x_min:x_max]


# ─────────────────────────────────────────────
# STEP 5 — RESIZE WITH PADDING
# ─────────────────────────────────────────────
def resize_with_padding(binary_img, target_h=TARGET_H, target_w=TARGET_W):
    """
    Resize the signature image to exactly (target_h x target_w) while
    preserving the original aspect ratio by padding with white.

    Why not just resize directly?
      - Direct resize would squish or stretch the signature
      - Shape distortion destroys one of the key features the model
        relies on — the proportions and stroke geometry of the signature

    Strategy:
      1. Compute scale factor so the longest side fits within target
      2. Resize using that scale (both dimensions scaled equally)
      3. Create a blank white canvas of the target size
      4. Center the resized signature on the canvas
      5. Result: signature floats centered in a white frame

    Input:  numpy array (H, W)          — binary image, any size
    Output: numpy array (target_h, target_w) — padded binary image
    """
    h, w = binary_img.shape

    # Compute scale factor — fit inside target without exceeding either dimension
    scale = min(target_w / w, target_h / h)

    # New dimensions after scaling (use int to get pixel counts)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize the image to the new dimensions
    # INTER_AREA is best for shrinking — preserves stroke quality
    resized = cv2.resize(binary_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a blank white canvas at the exact target size
    canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255

    # Compute offsets to center the resized image on the canvas
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2

    # Place the resized signature centered on the white canvas
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


# ─────────────────────────────────────────────
# STEP 6 — PIXEL NORMALIZATION
# ─────────────────────────────────────────────
def normalize(img):
    """
    Scale pixel values from [0, 255] to [0.0, 1.0] by dividing by 255.

    Why normalize:
      - Neural networks train much better on small values
      - Large raw pixel values (0-255) cause large activations
        which lead to unstable gradients during backpropagation
      - Values in [0, 1] keep activations in a stable range

    Note: standardization (mean/std centering) is a separate step
    done later across the whole dataset, not per image.

    Input:  numpy array (H, W) — uint8, values 0-255
    Output: numpy array (H, W) — float32, values 0.0-1.0
    """
    return img.astype(np.float32) / 255.0


# ─────────────────────────────────────────────
# STEP 7 — SAVE PROCESSED IMAGE
# ─────────────────────────────────────────────
def save_image(img_float, save_path):
    """
    Convert the normalized float image back to uint8 and save as PNG.

    Why PNG and not JPG for processed images:
      - PNG is lossless — no compression artifacts introduced
      - JPG compression would slightly alter pixel values we carefully
        normalized, adding noise to clean binary strokes
      - File size difference is negligible at 155x220

    Input:  numpy array (H, W) — float32, values 0.0-1.0
    Output: saves PNG file to save_path
    """
    # Convert float [0,1] back to uint8 [0,255] for saving
    img_uint8 = (img_float * 255).astype(np.uint8)
    cv2.imwrite(save_path, img_uint8)


# ─────────────────────────────────────────────
# FULL PIPELINE — SINGLE IMAGE
# ─────────────────────────────────────────────
def preprocess_image(img_path):
    """
    Run a single image through the full 7-step preprocessing pipeline.
    Returns the final normalized float32 numpy array ready for training.

    Returns None if the image cannot be read or processed,
    so the caller can skip and log it rather than crash.

    Input:  path to a raw image file
    Output: numpy array (TARGET_H, TARGET_W) — float32, [0,1]
            or None if processing failed
    """
    try:
        # Load the raw image and force RGB so later stages behave consistently.
        pil_img = Image.open(img_path).convert('RGB')
        img     = np.array(pil_img)

        # Apply the preprocessing stages in sequence.
        gray = to_grayscale(img)
        binary = binarize(gray)
        binary = ensure_white_background(binary)
        cropped = crop_to_content(binary)
        resized = resize_with_padding(cropped)
        normalized = normalize(resized)

        return normalized

    except Exception as e:
        print(f"  ❌ Failed to process {img_path}: {e}")
        return None


# ─────────────────────────────────────────────
# PROCESS ENTIRE DATASET
# ─────────────────────────────────────────────
def process_dataset(data_dir, processed_dir):
    """
    Walk through every subject in data_dir, apply the full pipeline
    to every image, and save results to processed_dir mirroring
    the exact same folder structure.

    Processed images are saved as PNG (lossless) regardless of
    the original format.

    data_dir structure expected:
        data/
          SIG_0001/
            genuine/  ← 10 images
            forged/   ← 5 images
          SIG_0002/
            ...

    Output structure (mirrors input):
        data_processed/
          SIG_0001/
            genuine/
            forged/
          ...
    """
    subjects = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('SIG_')
    ])

    print(f"Subjects to process: {len(subjects)}")
    print(f"Output directory:    {processed_dir}\n")

    total     = 0
    succeeded = 0
    failed    = []

    for subject in subjects:
        for label in ['genuine', 'forged']:
            src_folder = os.path.join(data_dir,      subject, label)
            dst_folder = os.path.join(processed_dir, subject, label)

            if not os.path.exists(src_folder):
                continue

            # Create matching output folder so the processed dataset mirrors the raw dataset.
            os.makedirs(dst_folder, exist_ok=True)

            img_files = sorted([
                f for f in os.listdir(src_folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])

            for fname in img_files:
                total    += 1
                src_path  = os.path.join(src_folder, fname)

                # Output always saved as PNG (lossless)
                dst_name  = os.path.splitext(fname)[0] + '.png'
                dst_path  = os.path.join(dst_folder, dst_name)

                # Skip if already processed, which lets the script resume safely.
                if os.path.exists(dst_path):
                    succeeded += 1
                    continue

                result = preprocess_image(src_path)

                if result is not None:
                    save_image(result, dst_path)
                    succeeded += 1
                else:
                    failed.append(src_path)

        # Progress per subject
        print(f"  ✅ {subject} done")

    # Summary
    print(f"\n{'=' * 50}")
    print(f"PREPROCESSING COMPLETE")
    print(f"{'=' * 50}")
    print(f"  Total images:     {total}")
    print(f"  Succeeded:        {succeeded}")
    print(f"  Failed:           {len(failed)}")

    if failed:
        print(f"\n  ⚠️  Failed files:")
        for f in failed:
            print(f"     {f}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == '__main__':
    # Make sure opencv is available
    try:
        import cv2
    except ImportError:
        print("OpenCV not found. Run: pip install opencv-python")
        exit(1)

    print("=" * 50)
    print("SIGNATURE PREPROCESSING PIPELINE")
    print("=" * 50)
    print(f"Source:  {DATA_DIR}")
    print(f"Output:  {PROCESSED_DIR}")
    print(f"Target size: {TARGET_H}h x {TARGET_W}w")
    print("=" * 50 + "\n")

    process_dataset(DATA_DIR, PROCESSED_DIR)