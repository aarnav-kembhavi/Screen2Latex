# Screen2LaTeX: A100 Training Guide

## 1. Setup on Server
```bash
git clone https://github.com/YOUR_USERNAME/screen2latex.git
cd screen2latex

pip install -r requirements.txt
```

## 2. Create WebDataset Shards
Download UniMER from HuggingFace and write tar shards (no individual image files).

```bash
# Writes data_ready/shards/shard-000000.tar, shard-000001.tar, ... (5000 samples each)
python create_shards.py --out-dir data_ready/shards
```

Optional: `--max-samples N` to limit samples, `--shard-size N` to change samples per shard (default 5000).

## 3. Vocab
Training requires `data_ready/vocab.txt` (one token per line). Build it from your formula corpus or use a pre-made LaTeX token vocab.

## 4. Run Training (A100)
Training uses only WebDataset shards. It will raise a clear error if `data_ready/shards/` is missing or empty.

```bash
tmux new -s training

python src/train.py --data-dir ./data_ready --batch-size 128 --epochs 10

# Detach: Ctrl+B, then D
# Re-attach: tmux attach -t training
```

Shards path: `--shards-dir` (default `data-dir/shards`). Checkpoints go to `--save-dir` (default `checkpoints`).

## 5. Inference
```bash
python src/inference.py path/to/image.png
```
Uses GPU by default if available. Requires `checkpoints/best_model.pt` and `vocab.txt` (e.g. `data_ready/vocab.txt`).

## Project Structure
```
Screen2Latex
├── src
│   ├── model.py
│   ├── train.py
│   ├── dataset.py
│   ├── inference.py
│   └── tokenizer.py
├── create_shards.py
├── data_ready
│   └── shards
└── checkpoints
```
