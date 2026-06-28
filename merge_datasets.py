"""
Merges two separately-downloaded Kaggle brain tumor MRI datasets into the
single data/Training/<class>/ and data/Testing/<class>/ structure that
data_pipeline.py expects.

Sources combined (14,176 images total):
  1. Masoud Nickparvar's "Brain Tumor MRI Dataset" (7,023 images)
  2. deeppythonist's brain tumor dataset (7,153 images, predefined train/test split)

Both use the same 4 classes: glioma, meningioma, notumor, pituitary — but
folder naming conventions can differ slightly between sources (e.g.
'glioma_tumor' vs 'glioma', or a nested extra folder level), so this script
normalizes both before copying.

Usage:
    python merge_datasets.py --source1 /path/to/nickparvar_extracted \\
                              --source2 /path/to/deeppythonist_extracted

Each --sourceN should point at the extracted folder that *contains*
Training/Testing (or equivalent) subfolders — the script searches a few
levels deep to find them, since Kaggle zip layouts vary.

Filename collisions across sources are handled by prefixing source2's
files with 's2_' before copying, so no image is silently overwritten.
"""
import argparse
import os
import shutil

from config import TRAIN_DIR, TEST_DIR, DATA_DIR, CLASS_NAMES

# Folder-name variants seen across different Kaggle re-uploads of this
# dataset family. Maps "what the source might call it" -> "our standard name".
CLASS_NAME_VARIANTS = {
    "glioma": "glioma",
    "glioma_tumor": "glioma",
    "meningioma": "meningioma",
    "meningioma_tumor": "meningioma",
    "notumor": "notumor",
    "no_tumor": "notumor",
    "pituitary": "pituitary",
    "pituitary_tumor": "pituitary",
}

SPLIT_NAME_VARIANTS = {
    "training": "Training",
    "train": "Training",
    "testing": "Testing",
    "test": "Testing",
    "val": "Testing",  # some sources call their holdout 'val' instead of 'Testing'
    "validation": "Testing",
}


def _find_split_dirs(root):
    """
    Walk `root` looking for directories that look like a Training/Testing
    split (case-insensitive, a few synonyms). Returns {split_name: path}.
    Searches up to 3 levels deep since Kaggle zips often nest an extra
    folder (e.g. root/Brain-Tumor-MRI/Training/...).
    """
    found = {}
    for dirpath, dirnames, _ in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if depth > 3:
            dirnames[:] = []  # stop descending further
            continue
        for d in dirnames:
            key = d.lower()
            if key in SPLIT_NAME_VARIANTS:
                standard = SPLIT_NAME_VARIANTS[key]
                # Prefer the first/shallowest match found.
                found.setdefault(standard, os.path.join(dirpath, d))
    return found


def _find_class_dirs(split_dir):
    """Return {standard_class_name: path} for class folders under a split dir."""
    found = {}
    if not os.path.isdir(split_dir):
        return found
    for d in os.listdir(split_dir):
        full = os.path.join(split_dir, d)
        if not os.path.isdir(full):
            continue
        key = d.lower()
        if key in CLASS_NAME_VARIANTS:
            found[CLASS_NAME_VARIANTS[key]] = full
    return found


def _copy_images(src_class_dir, dst_class_dir, prefix=""):
    os.makedirs(dst_class_dir, exist_ok=True)
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    count = 0
    for fname in os.listdir(src_class_dir):
        if not fname.lower().endswith(exts):
            continue
        src = os.path.join(src_class_dir, fname)
        dst_name = f"{prefix}{fname}" if prefix else fname
        dst = os.path.join(dst_class_dir, dst_name)
        # Avoid accidental overwrite even with a prefix, by disambiguating
        # further if a collision still somehow occurs.
        n = 1
        base, ext = os.path.splitext(dst)
        while os.path.exists(dst):
            dst = f"{base}_{n}{ext}"
            n += 1
        shutil.copy2(src, dst)
        count += 1
    return count


def merge_source(source_root, prefix, label):
    print(f"\n📂 Scanning {label}: {source_root}")
    if not os.path.exists(source_root):
        raise FileNotFoundError(f"{label} path does not exist: {source_root}")

    splits = _find_split_dirs(source_root)
    if not splits:
        raise ValueError(
            f"Could not find Training/Testing-style folders under {source_root}. "
            f"Check the extracted folder structure with: find {source_root} -maxdepth 3 -type d"
        )
    print(f"  Found splits: {list(splits.keys())}")

    totals = {}
    for split_key, split_dir in splits.items():
        dest_split_dir = TRAIN_DIR if split_key == "Training" else TEST_DIR
        class_dirs = _find_class_dirs(split_dir)
        missing = [c for c in CLASS_NAMES if c not in class_dirs]
        if missing:
            print(f"  ⚠️  {split_key}: missing class folders {missing} (found: {list(class_dirs.keys())})")

        for cls, cls_dir in class_dirs.items():
            dest_class_dir = os.path.join(dest_split_dir, cls)
            n = _copy_images(cls_dir, dest_class_dir, prefix=prefix)
            totals[f"{split_key}/{cls}"] = n
            print(f"  {split_key}/{cls}: copied {n} images")

    return totals


def main():
    parser = argparse.ArgumentParser(description="Merge two Kaggle brain tumor datasets into data/Training, data/Testing")
    parser.add_argument("--source1", required=True, help="Path to extracted Nickparvar dataset (or any first source)")
    parser.add_argument("--source2", required=True, help="Path to extracted deeppythonist dataset (or any second source)")
    parser.add_argument("--fresh", action="store_true", help="Delete existing data/Training and data/Testing before merging")
    args = parser.parse_args()

    if args.fresh:
        for d in (TRAIN_DIR, TEST_DIR):
            if os.path.exists(d):
                shutil.rmtree(d)
                print(f"🗑️  Removed existing {d}")

    os.makedirs(DATA_DIR, exist_ok=True)

    totals1 = merge_source(args.source1, prefix="", label="Source 1 (Nickparvar)")
    totals2 = merge_source(args.source2, prefix="s2_", label="Source 2 (deeppythonist)")

    print("\n" + "=" * 50)
    print("✅ MERGE COMPLETE")
    print("=" * 50)
    grand_total = 0
    for split_dir, split_name in [(TRAIN_DIR, "Training"), (TEST_DIR, "Testing")]:
        if not os.path.exists(split_dir):
            continue
        print(f"\n{split_name}:")
        for cls in CLASS_NAMES:
            cls_dir = os.path.join(split_dir, cls)
            if os.path.exists(cls_dir):
                n = len([f for f in os.listdir(cls_dir)
                         if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))])
                print(f"  {cls}: {n} images")
                grand_total += n
            else:
                print(f"  {cls}: 0 images (folder missing!)")
    print(f"\nGrand total: {grand_total} images")


if __name__ == "__main__":
    main()
