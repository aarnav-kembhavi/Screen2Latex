"""
Download UniMER dataset from HuggingFace and write WebDataset tar shards.
No individual image files are written; all data goes into data_ready/shards/*.tar.
"""

import io
import os
import argparse

from datasets import load_dataset
import webdataset as wds


SHARDS_DIR = "data_ready/shards"
SAMPLES_PER_SHARD = 10000


def main():
    p = argparse.ArgumentParser(description="Create WebDataset shards from UniMER (HuggingFace)")
    p.add_argument("--out-dir", type=str, default=SHARDS_DIR, help="Output directory for shard tars")
    p.add_argument("--max-samples", type=int, default=None, help="Max samples to write (default: all)")
    p.add_argument("--shard-size", type=int, default=SAMPLES_PER_SHARD, help="Samples per shard")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pattern = os.path.join(args.out_dir, "shard-%06d.tar")
    sink = wds.ShardWriter(pattern, maxcount=args.shard_size)

    print("Loading UniMER dataset (streaming)...")
    dataset = load_dataset("wanderkid/UniMER_Dataset", split="train", streaming=True)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x

    n = 0
    iterator = iter(dataset)
    pbar = tqdm(iterator, total=args.max_samples, desc="Writing shards", unit=" samples")
    for i, row in enumerate(pbar):
        if args.max_samples is not None and n >= args.max_samples:
            break
        img = row.get("image") or (row.get("images", [None])[0] if row.get("images") else None)
        label = row.get("label") or row.get("text", "") or ""
        if img is None:
            continue
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        image_bytes = buf.getvalue()
        latex_string = label if isinstance(label, str) else (label.decode("utf-8") if isinstance(label, bytes) else "")
        sample = {
            "__key__": str(n),
            "jpg": image_bytes,
            "txt": latex_string.encode("utf-8"),
        }
        sink.write(sample)
        n += 1

    sink.close()
    print(f"Wrote {n} samples to shards in {args.out_dir}")


if __name__ == "__main__":
    main()
