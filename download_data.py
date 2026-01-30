"""
Download UniMER dataset from Hugging Face (wanderkid/UniMER_Dataset).
Saves images to RAW_DIR/train/ and writes labels TSV (filename \\t formula).
"""
import os
import argparse

# Use datasets library for UniMER (HF Dataset, not raw repo)
from datasets import load_dataset

RAW_DIR = "./unimer_raw"
TRAIN_DIR = "train"
LABELS_FILE = "train.txt"


def main():
    p = argparse.ArgumentParser(description="Download UniMER dataset from Hugging Face")
    p.add_argument("--out-dir", type=str, default=RAW_DIR, help="Output root directory")
    p.add_argument("--max-samples", type=int, default=None, help="Max samples to download (default: all)")
    p.add_argument("--streaming", action="store_true", help="Stream dataset (saves memory for full 1M+)")
    args = p.parse_args()

    out_root = args.out_dir
    train_path = os.path.join(out_root, TRAIN_DIR)
    os.makedirs(train_path, exist_ok=True)
    labels_path = os.path.join(out_root, LABELS_FILE)

    print(f"Downloading UniMER dataset to {out_root}...")
    print("This may take a long time for the full 1M+ dataset...")

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x

    # Load dataset (train split; has 'image' and 'label' columns)
    # Use streaming for large dataset so we don't load 1M+ into memory
    use_streaming = args.streaming or (args.max_samples is not None and args.max_samples < 10000)
    if use_streaming:
        ds = load_dataset("wanderkid/UniMER_Dataset", split="train", streaming=True)
        total = args.max_samples
    else:
        ds = load_dataset("wanderkid/UniMER_Dataset", split="train")
        total = min(len(ds), args.max_samples) if args.max_samples else len(ds)

    n = 0
    with open(labels_path, "w", encoding="utf-8") as f_labels:
        iterator = iter(ds)
        pbar = tqdm(iterator, total=total, desc="Saving samples", unit=" samples")
        for row in pbar:
            if args.max_samples is not None and n >= args.max_samples:
                break
            # Dataset columns: image (PIL), label (LaTeX string)
            img = row.get("image") or row.get("images", [None])[0]
            label = row.get("label") or row.get("text", "") or ""
            if img is None:
                continue
            ext = "png" if getattr(img, "format", None) == "PNG" else "jpg"
            filename = f"{TRAIN_DIR}/{n}.{ext}"
            filepath = os.path.join(out_root, filename)
            img.save(filepath)
            f_labels.write(f"{filename}\t{label}\n")
            n += 1

    print(f"Download complete. Saved {n} samples to {out_root}")
    print(f"Labels file: {labels_path}")
    print(f"Image directory: {train_path}")


if __name__ == "__main__":
    main()
