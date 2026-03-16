"""
Create WebDataset shards from the UniMER dataset using HuggingFace streaming.

Optimizations:
- JPEG encoding (much faster than PNG)
- ThreadPoolExecutor for parallel encoding
- prefetch buffer to overlap download + encoding
- large shards to reduce filesystem overhead
"""

import io
import os
import argparse
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset
import webdataset as wds
from PIL import Image

SHARDS_DIR = "data_ready/shards"
SAMPLES_PER_SHARD = 50000
JPEG_QUALITY = 90


def encode_sample(index_row):
    """Convert a dataset sample to WebDataset format."""
    index, row = index_row

    img = row.get("image") or (row.get("images", [None])[0] if row.get("images") else None)
    label = row.get("label") or row.get("text", "") or ""

    if img is None:
        return None

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=False)

    image_bytes = buffer.getvalue()

    if isinstance(label, bytes):
        label = label.decode("utf-8")

    return {
        "__key__": str(index),
        "jpg": image_bytes,
        "txt": label.encode("utf-8"),
    }


def main():
    parser = argparse.ArgumentParser(description="Create WebDataset shards from UniMER")
    parser.add_argument("--out-dir", type=str, default=SHARDS_DIR)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=SAMPLES_PER_SHARD)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--prefetch", type=int, default=512)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    shard_pattern = os.path.join(args.out_dir, "shard-%06d.tar")

    sink = wds.ShardWriter(
        shard_pattern,
        maxcount=args.shard_size,
        compress=False
    )

    print("Streaming UniMER dataset from HuggingFace...")

    dataset = load_dataset(
        "wanderkid/UniMER_Dataset",
        split="train",
        streaming=True
    )

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x

    executor = ThreadPoolExecutor(max_workers=args.num_workers)

    futures = []
    dataset_iterator = iter(dataset)

    written = 0
    skipped = 0

    progress = tqdm(total=args.max_samples, desc="Writing shards", unit="samples")

    while True:

        while len(futures) < args.prefetch:
            try:
                row = next(dataset_iterator)
            except StopIteration:
                break

            futures.append(
                executor.submit(encode_sample, (written + len(futures), row))
            )

        if not futures:
            break

        future = futures.pop(0)
        result = future.result()

        if result is None:
            skipped += 1
        else:
            sink.write(result)
            written += 1
            progress.update(1)

        if args.max_samples and written >= args.max_samples:
            break

    progress.close()
    sink.close()

    print(f"Wrote {written} samples ({skipped} skipped) to {args.out_dir}")


if __name__ == "__main__":
    main()
