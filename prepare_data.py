"""
Prepare data from downloader output: read unimer_raw/train.txt, resize images to 128px height,
save to data_ready/images/, build vocab and new train.txt.
"""

import argparse
import os

try:
    from PIL import Image
except ImportError as e:
    raise ImportError("Pillow required. pip install Pillow") from e

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x

try:
    from src.tokenizer import LatexTokenizer
except ImportError:
    from tokenizer import LatexTokenizer


def main():
    p = argparse.ArgumentParser(description="Prepare data from unimer_raw for training")
    p.add_argument("--raw-dir", type=str, default="./unimer_raw", help="Raw data dir (has train.txt, train/)")
    p.add_argument("--out-dir", type=str, default="./data_ready", help="Output dir (images/, vocab.txt, train.txt)")
    p.add_argument("--height", type=int, default=128, help="Target image height in pixels")
    args = p.parse_args()

    raw_dir = args.raw_dir
    out_dir = args.out_dir
    labels_path = os.path.join(raw_dir, "train.txt")
    images_out_dir = os.path.join(out_dir, "images")
    vocab_path = os.path.join(out_dir, "vocab.txt")
    train_out_path = os.path.join(out_dir, "train.txt")

    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    os.makedirs(images_out_dir, exist_ok=True)

    # First pass: collect all formulas for vocab
    lines = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            rel_path, formula = parts[0].strip(), parts[1].strip()
            if rel_path and formula:
                lines.append((rel_path, formula))

    formulas = [f for _, f in lines]
    token2id = LatexTokenizer.build_vocab(formulas, special_first=True)
    tokenizer = LatexTokenizer(token2id=token2id)
    tokenizer.save_vocab(vocab_path)
    print(f"Vocab saved: {vocab_path} ({len(token2id)} tokens)")

    # Second pass: resize images and write new train.txt
    with open(train_out_path, "w", encoding="utf-8") as f_out:
        for rel_path, formula in tqdm(lines, desc="Resize images"):
            # Load image from raw_dir (e.g. raw_dir/train/0.jpg)
            img_path = os.path.join(raw_dir, rel_path)
            if not os.path.isfile(img_path):
                continue
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            new_h = args.height
            new_w = max(1, int(w * (new_h / h)))
            img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            # Save to out_dir/images/<basename> (e.g. 0.jpg)
            base_name = os.path.basename(rel_path)
            out_img_path = os.path.join(images_out_dir, base_name)
            img.save(out_img_path)
            # New labels line: images/0.jpg \t formula
            new_rel = os.path.join("images", base_name)
            f_out.write(new_rel + "\t" + formula + "\n")

    print(f"Resized {len(lines)} images to {images_out_dir}")
    print(f"Labels: {train_out_path}")


if __name__ == "__main__":
    main()
