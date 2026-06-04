import os
import re

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
DATA_DIR = os.path.abspath(DATA_DIR)


def rename_sig_folders(data_dir):
    """
    Rename all subject folders to SIG_XXXX format (zero-padded 4 digits).
    Sorts them by any trailing number found in the folder name.
    """
    print("=" * 50)
    print("STEP 1 — Renaming subject folders to SIG_XXXX")
    print("=" * 50)

    entries = [
        e for e in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, e))
    ]

    # Extract folders that have any number in their name
    numbered = []
    for entry in entries:
        match = re.search(r'(\d+)', entry)
        if match:
            numbered.append((int(match.group(1)), entry))

    # Sort by number
    numbered.sort(key=lambda x: x[0])

    renamed = 0
    skipped = 0

    for idx, (num, old_name) in enumerate(numbered, start=1):
        new_name = f"SIG_{num:04d}"
        old_path = os.path.join(data_dir, old_name)
        new_path = os.path.join(data_dir, new_name)

        if old_name == new_name:
            skipped += 1
            continue

        if os.path.exists(new_path):
            print(f"  ⚠️  Cannot rename '{old_name}' → '{new_name}': target already exists")
            skipped += 1
            continue

        os.rename(old_path, new_path)
        print(f"  ✅ '{old_name}' → '{new_name}'")
        renamed += 1

    print(f"\n  Done. Renamed: {renamed} | Skipped/Already correct: {skipped}\n")


def rename_label_folders(data_dir):
    """
    Inside each SIG_XXXX folder, rename any folder starting with
    'g' or 'G' to 'genuine' and any starting with 'f' or 'F' to 'forged'.
    """
    print("=" * 50)
    print("STEP 2 — Renaming label folders (genuine / forged)")
    print("=" * 50)

    subjects = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("SIG_")
    ])

    renamed = 0
    skipped = 0
    issues  = []

    for subject in subjects:
        subject_path = os.path.join(data_dir, subject)
        children = [
            c for c in os.listdir(subject_path)
            if os.path.isdir(os.path.join(subject_path, c))
        ]

        for child in children:
            child_lower = child.lower()
            child_path  = os.path.join(subject_path, child)

            if child_lower == 'genuine' or child_lower == 'forged':
                # Already correct name but may be wrong case
                correct = child_lower
            elif child_lower.startswith('g'):
                correct = 'genuine'
            elif child_lower.startswith('f'):
                correct = 'forged'
            else:
                issues.append(f"{subject}/{child}: cannot determine label (doesn't start with g or f)")
                skipped += 1
                continue

            new_path = os.path.join(subject_path, correct)

            if child == correct:
                skipped += 1
                continue

            if os.path.exists(new_path):
                issues.append(f"{subject}/{child}: target '{correct}' already exists")
                skipped += 1
                continue

            os.rename(child_path, new_path)
            print(f"  ✅ {subject}/{child} → {correct}")
            renamed += 1

    if issues:
        print(f"\n  ⚠️  Could not resolve:")
        for i in issues:
            print(f"     {i}")

    print(f"\n  Done. Renamed: {renamed} | Skipped/Already correct: {skipped}\n")


def fix_image_extensions(data_dir):
    """
    For every file inside genuine/ and forged/ folders:
    - If it has no extension → append .jpg
    - If it has an uppercase extension (.JPG, .PNG, .JPEG) → lowercase it
    - If it's already .jpg / .png / .jpeg → leave it alone
    """
    print("=" * 50)
    print("STEP 3 — Fixing image file extensions")
    print("=" * 50)

    subjects = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("SIG_")
    ])

    fixed    = 0
    skipped  = 0
    issues   = []

    valid_extensions = {'.jpg', '.jpeg', '.png'}

    for subject in subjects:
        subject_path = os.path.join(data_dir, subject)

        for label in ['genuine', 'forged']:
            label_path = os.path.join(subject_path, label)

            if not os.path.exists(label_path):
                continue

            files = [
                f for f in os.listdir(label_path)
                if os.path.isfile(os.path.join(label_path, f))
            ]

            for fname in files:
                fpath     = os.path.join(label_path, fname)
                name, ext = os.path.splitext(fname)

                # Case 1 — no extension at all
                if ext == '':
                    new_name = fname + '.jpg'
                    new_path = os.path.join(label_path, new_name)
                    if os.path.exists(new_path):
                        issues.append(f"{subject}/{label}/{fname}: target {new_name} already exists")
                        skipped += 1
                        continue
                    os.rename(fpath, new_path)
                    print(f"  ✅ {subject}/{label}/{fname} → {new_name}")
                    fixed += 1

                # Case 2 — uppercase or mixed case extension
                elif ext.lower() in valid_extensions and ext != ext.lower():
                    new_name = name + ext.lower()
                    new_path = os.path.join(label_path, new_name)
                    if os.path.exists(new_path):
                        issues.append(f"{subject}/{label}/{fname}: target {new_name} already exists")
                        skipped += 1
                        continue
                    os.rename(fpath, new_path)
                    print(f"  ✅ {subject}/{label}/{fname} → {new_name}")
                    fixed += 1

                # Case 3 — already correct
                elif ext.lower() in valid_extensions:
                    skipped += 1

                # Case 4 — unknown extension
                else:
                    issues.append(f"{subject}/{label}/{fname}: unknown extension '{ext}' — left untouched")
                    skipped += 1

    if issues:
        print(f"\n  ⚠️  Could not fix:")
        for i in issues:
            print(f"     {i}")

    print(f"\n  Done. Fixed: {fixed} | Skipped/Already correct: {skipped}\n")


def main():
    print(f"\nData directory: {DATA_DIR}\n")

    rename_sig_folders(DATA_DIR)
    rename_label_folders(DATA_DIR)
    fix_image_extensions(DATA_DIR)

    print("=" * 50)
    print("Structure fix complete. Run audit.py next.")
    print("=" * 50)


if __name__ == "__main__":
    main()