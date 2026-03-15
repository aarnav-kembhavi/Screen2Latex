"""
Prepare data from downloader output: read unimer_raw/train.txt, resize images to 128px height,
save to data_ready/images/ + train.txt, or write WebDataset tar shards.
"""

import argparse
import io
import os
import tarfile

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


def _pad_width(w: int, multiple: int = 32) -> int:
    return ((w + multiple - 1) // multiple) * multiple


def main():
    p = argparse.ArgumentParser(description="Prepare data from unimer_raw for training")
    p.add_argument("--raw-dir", type=str, default="./unimer_raw", help="Raw data dir (has train.txt, train/)")
    p.add_argument("--out-dir", type=str, default="./data_ready", help="Output dir (images/, vocab.txt, train.txt or shards/)")
    p.add_argument("--height", type=int, default=128, help="Target image height in pixels")
    p.add_argument("--shard-size", type=int, default=10000, help="Samples per tar shard (WebDataset)")
    p.add_argument("--output-format", type=str, default="files", choices=("files", "webdataset"),
                   help="Output: 'files' (images/ + train.txt) or 'webdataset' (tar shards)")
    args = p.parse_args()

    raw_dir = args.raw_dir
    out_dir = args.out_dir
    labels_path = os.path.join(raw_dir, "train.txt")
    vocab_path = os.path.join(out_dir, "vocab.txt")

    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

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

    if args.output_format == "webdataset":
        # Write tar shards: data_ready/shards/unimer-000000.tar, ...
        shards_dir = os.path.join(out_dir, "shards")
        os.makedirs(shards_dir, exist_ok=True)
        shard_size = args.shard_size
        total = 0
        shard_idx = 0
        current_tar_path = None
        current_tar = None
        for i, (rel_path, formula) in enumerate(tqdm(lines, desc="Writing shards")):
            if current_tar is None or (total > 0 and total % shard_size == 0):
                if current_tar is not None:
                    current_tar.close()
                current_tar_path = os.path.join(shards_dir, f"unimer-{shard_idx:06d}.tar")
                current_tar = tarfile.open(current_tar_path, "w")
                shard_idx += 1
            img_path = os.path.join(raw_dir, rel_path)
            if not os.path.isfile(img_path):
                continue
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            new_h = args.height
            new_w = max(1, int(w * (new_h / h)))
            new_w = _pad_width(new_w, 32)
            img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            image_bytes = buf.getvalue()
            key = f"{total:09d}"
            info = tarfile.TarInfo(name=f"{key}.jpg")
            info.size = len(image_bytes)
            current_tar.addfile(info, io.BytesIO(image_bytes))
            label_bytes = formula.encode("utf-8")
            info = tarfile.TarInfo(name=f"{key}.txt")
            info.size = len(label_bytes)
            current_tar.addfile(info, io.BytesIO(label_bytes))
            total += 1
        if current_tar is not None:
            current_tar.close()
        print(f"Wrote {total} samples to {shard_idx} shards in {shards_dir}")
        return

    # Legacy: images/ + train.txt
    images_out_dir = os.path.join(out_dir, "images")
    train_out_path = os.path.join(out_dir, "train.txt")
    os.makedirs(images_out_dir, exist_ok=True)
    with open(train_out_path, "w", encoding="utf-8") as f_out:
        for rel_path, formula in tqdm(lines, desc="Resize images"):
            img_path = os.path.join(raw_dir, rel_path)
            if not os.path.isfile(img_path):
                continue
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            new_h = args.height
            new_w = max(1, int(w * (new_h / h)))
            img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            base_name = os.path.basename(rel_path)
            out_img_path = os.path.join(images_out_dir, base_name)
            img.save(out_img_path)
            new_rel = os.path.join("images", base_name)
            f_out.write(new_rel + "\t" + formula + "\n")
    print(f"Resized {len(lines)} images to {images_out_dir}")
    print(f"Labels: {train_out_path}")


if __name__ == "__main__":
    main()
