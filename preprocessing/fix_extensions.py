import os
import pillow_heif
from PIL import Image

pillow_heif.register_heif_opener()

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

HEIC_BRANDS = {b'heic', b'heix', b'hevc', b'hevx', b'mif1', b'msf1'}

FORMAT_MAP = {
    'jpeg': '.jpg',
    'png':  '.png',
    'bmp':  '.bmp',
    'webp': '.webp',
    'gif':  '.gif',
    'tiff': '.tiff',
    'heif': '.jpg',  # HEIC/HEIF always converts to jpg
}


def detect_real_format(fpath):
    """
    Detects real image format purely from file header bytes.
    Never trusts the file extension.
    """
    try:
        with open(fpath, 'rb') as f:
            header = f.read(12)

        if len(header) < 4:
            return None

        # HEIC — box type at 4:8 is 'ftyp', brand at 8:12
        if len(header) >= 12 and header[4:8] == b'ftyp' and header[8:12] in HEIC_BRANDS:
            return 'heic'

        # JPEG
        if header[:3] == b'\xff\xd8\xff':
            return 'jpeg'

        # PNG
        if header[:4] == b'\x89PNG':
            return 'png'

        # BMP
        if header[:2] == b'BM':
            return 'bmp'

        # WEBP
        if header[:4] == b'RIFF' and header[8:12] == b'WEBP':
            return 'webp'

        # TIFF
        if header[:4] in (b'II*\x00', b'MM\x00*'):
            return 'tiff'

        return None

    except Exception:
        return None


def convert_to_jpg(src_path, dst_path):
    """
    Convert any image to JPEG using Pillow + pillow_heif.
    Returns True on success, False on failure.
    """
    try:
        img = Image.open(src_path).convert('RGB')
        img.save(dst_path, 'JPEG', quality=95)
        return True
    except Exception as e:
        print(f"  ❌ Conversion failed for {src_path}: {e}")
        return False


def fix_extensions(data_dir):
    """
    Walk every file in every subject folder.
    Detect real format from bytes, convert/rename accordingly.
    HEIC files are converted to JPEG in-place.
    Wrong extensions are corrected.
    """
    print("=" * 50)
    print("Fixing image extensions from raw byte headers")
    print("=" * 50)

    # Collect all files first regardless of folder name
    # (raw data may still have Genuine/Forge/geniue etc.)
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for fname in files:
            all_files.append(os.path.join(root, fname))

    converted = 0
    renamed   = 0
    already   = 0
    failed    = []

    for fpath in sorted(all_files):
        # Skip hidden or system files
        fname = os.path.basename(fpath)
        if fname.startswith('.'):
            continue

        real_format = detect_real_format(fpath)
        name, ext   = os.path.splitext(fname)
        folder      = os.path.dirname(fpath)

        # Undetectable
        if real_format is None:
            failed.append(f"{fpath} (undetectable)")
            continue

        # HEIC — convert to JPEG, delete original
        if real_format == 'heic':
            dst_path = os.path.join(folder, name + '.jpg')

            # If same path (no extension case), use temp name to avoid collision
            if fpath == dst_path:
                dst_path = os.path.join(folder, name + '_converted.jpg')

            success = convert_to_jpg(fpath, dst_path)
            if success:
                os.remove(fpath)
                # Rename _converted back to clean name if needed
                final_path = os.path.join(folder, name + '.jpg')
                if dst_path != final_path and not os.path.exists(final_path):
                    os.rename(dst_path, final_path)
                rel = os.path.relpath(fpath, data_dir)
                print(f"  🔄 {rel} → {name}.jpg (HEIC→JPEG)")
                converted += 1
            else:
                failed.append(fpath)
            continue

        # All other formats — just fix extension if wrong
        correct_ext = FORMAT_MAP.get(real_format)
        if correct_ext is None:
            failed.append(f"{fpath} (unsupported: {real_format})")
            continue

        if ext.lower() == correct_ext:
            already += 1
            continue

        new_path = os.path.join(folder, name + correct_ext)
        if os.path.exists(new_path):
            failed.append(f"{fpath}: target already exists")
            continue

        os.rename(fpath, new_path)
        rel = os.path.relpath(fpath, data_dir)
        print(f"  ✅ {rel} → {name}{correct_ext}")
        renamed += 1

    print(f"\n  Converted (HEIC→JPG): {converted}")
    print(f"  Renamed (wrong ext):  {renamed}")
    print(f"  Already correct:      {already}")

    if failed:
        print(f"\n  ⚠️  Failed ({len(failed)}):")
        for f in failed[:20]:
            print(f"     {f}")
        if len(failed) > 20:
            print(f"     ... and {len(failed) - 20} more")

    print()


if __name__ == "__main__":
    fix_extensions(DATA_DIR)
    print("=" * 50)
    print("Done. Run fix_structure.py then audit.py next.")
    print("=" * 50)