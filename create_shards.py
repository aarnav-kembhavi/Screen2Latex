"""
Create WebDataset shards from UniMER dataset.
Saves images as pre-decoded, pre-resized numpy arrays (.npy) so training
requires zero PIL decode or resize — just load and normalize.
"""

import io
import os
import sys
import json
import hashlib
import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import webdataset as wds
from PIL import Image

INPUT_DIR = "unimer"
SHARDS_DIR = "data_ready/shards"
SAMPLES_PER_SHARD = 10000
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
TRAIN_TXT = "train.txt"

IMG_HEIGHT = 128
MAX_WIDTH = 1024
WIDTH_MULTIPLE = 32


def pad_width_to_multiple(w: int, multiple: int = WIDTH_MULTIPLE) -> int:
    return ((w + multiple - 1) // multiple) * multiple


def collect_dataset(root_dir: str) -> List[Tuple[str, str]]:
    """Collect (image_path, label) pairs from train.txt or sidecar files."""
    root_dir = os.path.abspath(root_dir)
    samples: List[Tuple[str, str]] = []

    train_txt = os.path.join(root_dir, TRAIN_TXT)
    if os.path.isfile(train_txt):
        with open(train_txt, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        first_line = lines[0]
        parts = first_line.split(maxsplit=1)
        is_path_label = (
            len(parts) == 2
            and any(parts[0].endswith(ext) for ext in [".png", ".jpg", ".jpeg"])
        )

        if is_path_label:
            print("Detected train.txt format: path + label")
            for line in lines:
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                rel_path, label = parts[0].strip(), parts[1].strip()
                if not label or len(label) < 3 or len(label) > 512:
                    continue
                try:
                    label.encode("utf-8")
                except (UnicodeEncodeError, UnicodeDecodeError):
                    continue
                img_path = os.path.join(root_dir, rel_path)
                if not os.path.isfile(img_path):
                    img_path = os.path.join(root_dir, "images", rel_path)
                if os.path.isfile(img_path):
                    samples.append((img_path, label))
        else:
            print("Detected train.txt format: label-only → index alignment")
            images_dir = os.path.join(root_dir, "images")
            if not os.path.isdir(images_dir):
                raise RuntimeError("images/ directory not found")
            image_files = sorted([
                f for f in os.listdir(images_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])
            n = min(len(image_files), len(lines))
            for i in range(n):
                img_path = os.path.join(images_dir, image_files[i])
                label = lines[i]
                if not label or len(label) < 3 or len(label) > 512:
                    continue
                try:
                    label.encode("utf-8")
                except (UnicodeEncodeError, UnicodeDecodeError):
                    continue
                samples.append((img_path, label))

        print(f"Found {len(samples)} samples (from {TRAIN_TXT})")
        return samples

    # Fallback: sidecar .json/.txt
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            img_path = os.path.join(dirpath, fname)
            base = os.path.splitext(img_path)[0]
            label = ""
            json_path = base + ".json"
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    label = meta.get("label") or meta.get("text") or ""
                except Exception:
                    label = ""
            if not label:
                txt_path = base + ".txt"
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, "r", encoding="utf-8") as f:
                            label = f.read().strip()
                    except Exception:
                        label = ""
            if not isinstance(label, str) or not label or len(label) < 3 or len(label) > 2000:
                continue
            try:
                label.encode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            samples.append((img_path, label))

    print(f"Found {len(samples)} samples (sidecar labels)")
    return samples


def encode_sample(path_label: Tuple[str, str]):
    """
    Convert (path, label) to WebDataset sample with pre-processed numpy image.
    Image is resized to IMG_HEIGHT, width capped at MAX_WIDTH, saved as .npy
    """
    path, label = path_label
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            w, h = img.size
            new_w = max(1, int(w * (IMG_HEIGHT / h)))
            new_w = min(new_w, MAX_WIDTH)
            new_w = ((new_w + WIDTH_MULTIPLE - 1) // WIDTH_MULTIPLE) * WIDTH_MULTIPLE
            img = img.resize((new_w, IMG_HEIGHT), Image.Resampling.BILINEAR)
            arr = np.array(img, dtype=np.uint8)  # (H, W, 3)

        buf = io.BytesIO()
        np.save(buf, arr)
        npy_bytes = buf.getvalue()

    except Exception as e:
        print(f"Warning: {e}")
        return None

    if isinstance(label, bytes):
        label = label.decode("utf-8")

    key = hashlib.md5(path.encode("utf-8")).hexdigest()
    return {
        "__key__": key,
        "npy": npy_bytes,
        "txt": label.encode("utf-8"),
    }


def main():
    parser = argparse.ArgumentParser(description="Create numpy WebDataset shards from UniMER")
    parser.add_argument("--root-dir", type=str, default=INPUT_DIR)
    parser.add_argument("--out-dir", type=str, default=SHARDS_DIR)
    parser.add_argument("--shard-size", type=int, default=SAMPLES_PER_SHARD)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Scanning dataset...")
    samples = collect_dataset(args.root_dir)
    if not samples:
        raise RuntimeError(f"No image/label pairs found under {args.root_dir}")
    if args.limit is not None:
        samples = samples[:args.limit]
        print(f"Limited to {len(samples)} samples")

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x

    shard_pattern = os.path.join(args.out_dir, "shard-%06d.tar")
    sink = wds.ShardWriter(shard_pattern, maxcount=args.shard_size, compress=False)
    executor = ThreadPoolExecutor(max_workers=args.num_workers)

    total = len(samples)
    progress = tqdm(total=total, desc="Writing shards", unit="samples")
    pending = []
    iterator = iter(samples)
    written = 0
    skipped = 0

    while True:
        while len(pending) < args.prefetch:
            try:
                item = next(iterator)
            except StopIteration:
                break
            pending.append(executor.submit(encode_sample, item))

        if not pending:
            break

        future = pending.pop(0)
        try:
            result = future.result()
            if result is None:
                skipped += 1
            else:
                sink.write(result)
                written += 1
        except Exception as e:
            print(f"Warning: {e}")
            skipped += 1
        finally:
            progress.update(1)

    progress.close()
    sink.close()
    executor.shutdown(wait=True)
    print(f"Wrote {written} samples, skipped {skipped}. Shards in {args.out_dir}")


if __name__ == "__main__":
    main()