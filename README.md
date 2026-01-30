# Screen2LaTeX: A100 Training Guide

## 1. Setup on Server
```bash
# Clone the repo (Use your PAT if private)
git clone https://github.com/YOUR_USERNAME/screen2latex.git
cd screen2latex

# Install dependencies
pip install -r requirements.txt
```

## 2. Download Data
The server has fast internet. Download the UniMER-1M dataset directly here.

```bash
# Downloads images to ./raw_data
python download_data.py --out-dir ./raw_data
```

## 3. Prepare Data
Resize images to 128px height using the server CPU.

```bash
# Processes ./raw_data -> ./final_data
python prepare_data.py --raw-dir ./raw_data --out-dir ./final_data
```

## 4. Run Training (A100)
Run inside tmux to prevent disconnects.

```bash
tmux new -s training

# Start Training (Adjust batch size if needed, A100 supports 128+)
python src/train.py --data-dir ./final_data --batch-size 128 --epochs 10

# Detach: Ctrl+B, then D
# Re-attach: tmux attach -t training
```
