import os
import pandas as pd
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
DATA_DIR = os.path.abspath(DATA_DIR)

def run_audit(data_dir):
    subjects = sorted([
        f for f in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, f))
        and f.startswith("SIG_")
    ])

    print(f"Subjects found: {len(subjects)}\n")

    results = []
    issues  = []

    for subject in subjects:
        subject_path = os.path.join(data_dir, subject)
        genuine_path = os.path.join(subject_path, "genuine")
        forged_path  = os.path.join(subject_path, "forged")

        # Check folders exist
        if not os.path.exists(genuine_path):
            issues.append(f"{subject}: missing 'genuine' folder")
            continue
        if not os.path.exists(forged_path):
            issues.append(f"{subject}: missing 'forged' folder")
            continue

        genuine_imgs = [
            f for f in os.listdir(genuine_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        forged_imgs = [
            f for f in os.listdir(forged_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        # Check counts
        if len(genuine_imgs) != 10:
            issues.append(
                f"{subject}: expected 10 genuine, found {len(genuine_imgs)}"
            )
        if len(forged_imgs) != 5:
            issues.append(
                f"{subject}: expected 5 forged,   found {len(forged_imgs)}"
            )

        # Read each image
        for label, img_list, folder in [
            ("genuine", genuine_imgs, genuine_path),
            ("forged",  forged_imgs,  forged_path)
        ]:
            for img_file in img_list:
                img_path = os.path.join(folder, img_file)
                try:
                    img = Image.open(img_path)
                    w, h = img.size
                    mode = img.mode
                    results.append({
                        "subject": subject,
                        "label":   label,
                        "file":    img_file,
                        "width":   w,
                        "height":  h,
                        "mode":    mode
                    })
                except Exception as e:
                    issues.append(f"{subject}/{img_file}: CORRUPT — {e}")

    # Results
    df = pd.DataFrame(results)

    print("=" * 40)
    print("IMAGE DIMENSIONS")
    print("=" * 40)
    print(f"Width  — min: {df['width'].min():<6} max: {df['width'].max():<6} mean: {df['width'].mean():.1f}")
    print(f"Height — min: {df['height'].min():<6} max: {df['height'].max():<6} mean: {df['height'].mean():.1f}")

    print("\n" + "=" * 40)
    print("COLOR MODES")
    print("=" * 40)
    print(df['mode'].value_counts().to_string())

    print("\n" + "=" * 40)
    print("IMAGE COUNTS")
    print("=" * 40)
    print(df['label'].value_counts().to_string())

    print("\n" + "=" * 40)
    print(f"ISSUES ({len(issues)} found)")
    print("=" * 40)
    if issues:
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print("  ✅ No issues found")

    print("\n" + "=" * 40)
    print("SAMPLE ROWS")
    print("=" * 40)
    print(df.head(10).to_string(index=False))

    return df, issues

if __name__ == "__main__":
    df, issues = run_audit(DATA_DIR)