"""
A100-optimized training loop for Screen2LaTeX.
BFloat16 / AMP, high batch size, checkpoint every 5 epochs.
"""

import argparse
import os
from functools import partial
from pathlib import Path
from typing import Optional

try:
    import torch
    import torch.nn as nn
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.data import DataLoader
except ImportError as e:
    raise ImportError("PyTorch is required. Install with: pip install torch") from e

try:
    from src.model import MobileOneStudent
    from src.dataset import MathDataset, collate_fn_pad
    from src.tokenizer import LatexTokenizer
except ImportError:
    from model import MobileOneStudent
    from dataset import MathDataset, collate_fn_pad
    from tokenizer import LatexTokenizer


def _get_device() -> torch.device:
    """Prefer CUDA (A100), then MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(
    labels_path: Optional[str],
    image_dir: Optional[str],
    vocab_path: str,
    root_dir: Optional[str] = None,
    save_dir: str = "checkpoints",
    batch_size: int = 32,
    max_epochs: int = 100,
    max_len: int = 128,
    lr: float = 1e-4,
    checkpoint_every: int = 5,
    use_amp: bool = True,
    num_workers: int = 4,
) -> None:
    """
    Production training: TSV labels, real images, AMP, checkpoint every N epochs.
    Logits and targets flattened for CrossEntropyLoss; padding ignored via ignore_index.
    """
    device = _get_device()
    # A100: prefer bfloat16 if available; else fp16 via amp
    use_bfloat16 = use_amp and device.type == "cuda" and torch.cuda.is_bf16_supported()
    if device.type == "cuda" and use_amp and not use_bfloat16:
        torch.set_float32_matmul_precision("high")
    print(f"Device: {device}, AMP: {use_amp}, BF16: {use_bfloat16}")

    # Tokenizer and vocab
    tokenizer = LatexTokenizer(vocab_path=vocab_path)
    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_id

    # Dataset and loader: root_dir = data dir (train.txt, images/) or explicit labels_path + image_dir
    dataset = MathDataset(
        root_dir=root_dir,
        labels_path=labels_path,
        image_dir=image_dir,
        tokenizer=tokenizer,
        img_height=128,
        width_multiple=32,
        max_len=max_len,
        add_sos_eos=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=partial(collate_fn_pad, pad_id=pad_id),
        pin_memory=(device.type == "cuda"),
    )

    # Model and optimizer
    model = MobileOneStudent(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    scaler = GradScaler() if (use_amp and device.type == "cuda") else None

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(1, max_epochs + 1):
        running_loss = 0.0
        num_batches = 0
        for images, labels, _ in loader:
            # images: [B, 3, H, W], labels: [B, L]
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            # Forward: optional AMP (BF16 or FP16)
            if use_amp and device.type == "cuda":
                with autocast(dtype=torch.bfloat16 if use_bfloat16 else torch.float16):
                    logits = model(images)
                    # logits: [B, Seq_Len, vocab_size]; Seq_Len from CNN width (variable per batch)
                    seq_len = min(logits.size(1), labels.size(1))
                    # Flatten to [B*Seq_Len, vocab_size] and [B*Seq_Len] for CrossEntropyLoss
                    logits_flat = logits[:, :seq_len].reshape(-1, vocab_size)
                    labels_flat = labels[:, :seq_len].reshape(-1)
                    loss = criterion(logits_flat, labels_flat)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                seq_len = min(logits.size(1), labels.size(1))
                logits_flat = logits[:, :seq_len].reshape(-1, vocab_size)
                labels_flat = labels[:, :seq_len].reshape(-1)
                loss = criterion(logits_flat, labels_flat)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        mean_loss = running_loss / max(1, num_batches)
        print(f"Epoch {epoch}/{max_epochs} Loss: {mean_loss:.4f}")

        # Save checkpoint every N epochs
        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            ckpt_file = save_path / f"screen2latex_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "vocab_size": vocab_size,
                },
                ckpt_file,
            )
            print(f"Checkpoint saved: {ckpt_file}")


def main():
    p = argparse.ArgumentParser(description="Screen2LaTeX production training (A100-optimized)")
    p.add_argument("--data-dir", type=str, default="./data_ready", help="Data dir (train.txt, images/, vocab.txt)")
    p.add_argument("--labels", type=str, default=None, help="Path to TSV (overrides data-dir/train.txt)")
    p.add_argument("--images", type=str, default=None, help="Directory of images (overrides data-dir)")
    p.add_argument("--vocab", type=str, default=None, help="Path to vocab.txt (overrides data-dir/vocab.txt)")
    p.add_argument("--save-dir", type=str, default="checkpoints", help="Checkpoint directory")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p.add_argument("--epochs", type=int, default=100, help="Max epochs")
    p.add_argument("--max-len", type=int, default=128, help="Max sequence length")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--checkpoint-every", type=int, default=5, help="Save checkpoint every N epochs")
    p.add_argument("--no-amp", action="store_true", help="Disable AMP")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    labels_path = args.labels or str(data_dir / "train.txt")
    image_dir = args.images or str(data_dir)
    vocab_path = args.vocab or str(data_dir / "vocab.txt")

    # If using data-dir only, pass root_dir so dataset uses data_dir/train.txt and data_dir/images/
    if args.labels is None and args.images is None:
        labels_path = None
        image_dir = None
        root_dir = str(data_dir)
    else:
        root_dir = None

    train(
        labels_path=labels_path,
        image_dir=image_dir,
        vocab_path=vocab_path,
        root_dir=root_dir,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        max_len=args.max_len,
        lr=args.lr,
        checkpoint_every=args.checkpoint_every,
        use_amp=not args.no_amp,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
