"""
Sanity pipeline: create shards (small subset), build vocab if missing, run 1 epoch, run inference.
Run from repo root: python debug_run.py [root_dir] [image_path]
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)

    root_dir = sys.argv[1] if len(sys.argv) > 1 else "unimer"
    limit = 2000
    data_dir = "data_ready"
    shards_dir = f"{data_dir}/shards"
    vocab_path = f"{data_dir}/vocab.txt"

    print("=== 1. Create shards (small subset) ===")
    if not Path(shards_dir).exists() or not list(Path(shards_dir).glob("*.tar")):
        r = subprocess.run(
            [sys.executable, "create_shards.py", "--root-dir", root_dir, "--out-dir", shards_dir, "--limit", str(limit)],
            cwd=root,
        )
        if r.returncode != 0:
            print("create_shards.py failed")
            sys.exit(1)
        print("Shards created.")
    else:
        print("Shards already present, skipping.")

    print("=== 2. Train (1 epoch, debug) ===")
    r = subprocess.run(
        [
            sys.executable, "src/train.py",
            "--data-dir", data_dir,
            "--epochs", "1",
            "--debug",
        ],
        cwd=root,
    )
    if r.returncode != 0:
        print("train.py failed")
        sys.exit(1)
    print("Training done.")

    print("=== 3. Inference on one image ===")
    image_path = sys.argv[2] if len(sys.argv) > 2 else None
    if not image_path or not os.path.isfile(image_path):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            from PIL import Image
            img = Image.new("RGB", (200, 64), color=(255, 255, 255))
            img.save(f.name)
            image_path = f.name
        print(f"No image given; using dummy image: {image_path}")

    from src.inference import LatexPredictor, _latest_checkpoint
    ckpt = _latest_checkpoint("checkpoints")
    if not ckpt or not os.path.isfile(ckpt):
        print("No checkpoint found; skipping inference.")
        return

    predictor = LatexPredictor(ckpt, vocab_path)
    out = predictor.predict(image_path)
    print(f"Inference: {out}")
    print("=== Sanity pipeline done ===")


if __name__ == "__main__":
    main()
