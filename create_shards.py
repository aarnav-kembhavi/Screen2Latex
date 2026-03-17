"""
Create WebDataset shards from a local UniMER dataset directory.
Supports: train.txt (path + label per line) or sidecar .json/.txt per image.
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

import webdataset as wds
from PIL import Image

INPUT_DIR = "unimer"
SHARDS_DIR = "data_ready/shards"
SAMPLES_PER_SHARD = 50000
JPEG_QUALITY = 90
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
TRAIN_TXT = "train.txt"


def collect_dataset(root_dir: str) -> List[Tuple[str, str]]:
    """Collect (image_path, label) pairs. Prefer train.txt if present; else sidecar .json/.txt."""
    root_dir = os.path.abspath(root_dir)
    samples: List[Tuple[str, str]] = []

    train_txt = os.path.join(root_dir, TRAIN_TXT)
    if os.path.isfile(train_txt):
        with open(train_txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
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
                if os.path.isfile(img_path):
                    samples.append((img_path, label))
        print(f"Found {len(samples)} samples (from {TRAIN_TXT})")
        return samples

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
                except Exception as e:
                    print("Warning:", e)
                    label = ""

            if not label:
                txt_path = base + ".txt"
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, "r", encoding="utf-8") as f:
                            label = f.read().strip()
                    except Exception as e:
                        print("Warning:", e)
                        label = ""

            if not isinstance(label, str) or not label or len(label) < 3 or len(label) > 512:
                continue
            try:
                label.encode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            samples.append((img_path, label))

    print(f"Found {len(samples)} samples (sidecar labels)")
    return samples


def encode_sample(path_label: Tuple[str, str]):
    """Convert (path, label) to WebDataset sample dict."""
    path, label = path_label
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=False)
        image_bytes = buffer.getvalue()
    except Exception as e:
        print("Warning:", e)
        return None

    if isinstance(label, bytes):
        label = label.decode("utf-8")
    key = hashlib.md5(path.encode("utf-8")).hexdigest()
    return {
        "__key__": key,
        "jpg": image_bytes,
        "txt": label.encode("utf-8"),
    }


def main():
    parser = argparse.ArgumentParser(description="Create WebDataset shards from local UniMER dataset")
    parser.add_argument("--root-dir", type=str, default=INPUT_DIR, help="Root directory of dataset")
    parser.add_argument("--out-dir", type=str, default=SHARDS_DIR, help="Output directory for shard tars")
    parser.add_argument("--shard-size", type=int, default=SAMPLES_PER_SHARD, help="Samples per shard")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N samples")
    parser.add_argument("--num-workers", type=int, default=16, help="ThreadPoolExecutor workers")
    parser.add_argument("--prefetch", type=int, default=512, help="In-flight encoding tasks")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Scanning dataset...")
    samples = collect_dataset(args.root_dir)
    if not samples:
        raise RuntimeError(f"No image/label pairs found under {args.root_dir}")

    if args.limit is not None:
        samples = samples[: args.limit]
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
            print("Warning:", e)
            skipped += 1
        finally:
            progress.update(1)

    progress.close()
    sink.close()
    executor.shutdown(wait=True)
    print(f"Wrote {written} samples, skipped {skipped}. Shards in {args.out_dir}")


if __name__ == "__main__":
    main()
