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
print("RUNNING:", __file__)
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
            lines = [line.strip() for line in f if line.strip()]

        # Detect format
        first_line = lines[0]
        parts = first_line.split(maxsplit=1)

        is_path_label = (
            len(parts) == 2
            and (
                parts[0].endswith(".png")
                or parts[0].endswith(".jpg")
                or parts[0].endswith(".jpeg")
            )
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
            print("Detected train.txt format: label-only → using index alignment")

            images_dir = os.path.join(root_dir, "images")
            if not os.path.isdir(images_dir):
                raise RuntimeError("images/ directory not found for label-only dataset")

            image_files = sorted(
                [
                    f
                    for f in os.listdir(images_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
            )

            n = min(len(image_files), len(lines))

            if n == 0:
                raise RuntimeError("No valid image/label pairs after alignment")

            print(f"Aligning {n} samples (images ↔ labels)")

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
    """Convert (path, label) to WebDataset sample dict."""
    path, label = path_label
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            MAX_WIDTH = 2048  # try 1024 if still OOM
            print("BEFORE:", img.width, flush=True)

            if img.width > MAX_WIDTH:
                scale = MAX_WIDTH / img.width
                new_height = int(img.height * scale)
                img = img.resize((MAX_WIDTH, new_height))

            print("AFTER:", img.width, flush=True)
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
