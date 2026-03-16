"""
A100-optimized training loop for Screen2LaTeX.
BFloat16 / AMP, high batch size, checkpoint every 5 epochs.
Training requires WebDataset shards in data_ready/shards/.
"""

import argparse
from functools import partial
from pathlib import Path
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.data import DataLoader
except ImportError as e:
    raise ImportError("PyTorch is required. Install with: pip install torch") from e

try:
    from src.model import MobileOneStudent
    from src.dataset import WebFormulaDataset, collate_fn_pad
    from src.tokenizer import LatexTokenizer
except ImportError:
    from model import MobileOneStudent
    from dataset import WebFormulaDataset, collate_fn_pad
    from tokenizer import LatexTokenizer

# Stage -> max samples for progressive scaling
STAGE_MAX_SAMPLES = {"stage1": 50_000, "stage2": 200_000, "stage3": 500_000, "stage4": None}

SHARDS_DIR_DEFAULT = "data_ready/shards"


def _get_device() -> torch.device:
    """Prefer CUDA (A100), then MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(
    vocab_path: str,
    save_dir: str = "checkpoints",
    shards_dir: str = SHARDS_DIR_DEFAULT,
    batch_size: int = 96,
    max_epochs: int = 100,
    max_len: int = 128,
    lr: float = 1e-4,
    checkpoint_every: int = 5,
    use_amp: bool = True,
    num_workers: int = 8,
    teacher_checkpoint: Optional[str] = None,
    distill_alpha: float = 0.5,
    distill_temp: float = 2.0,
    max_samples: Optional[int] = None,
    stage: Optional[str] = None,
    freeze_backbone: bool = False,
    unfreeze_epoch: Optional[int] = None,
    curriculum: bool = False,
) -> None:
    """
    Production training: WebDataset shards only. AMP, checkpoint every N epochs.
    Logits and targets flattened for CrossEntropyLoss; padding ignored via ignore_index.
    """
    device = _get_device()
    use_bfloat16 = use_amp and device.type == "cuda" and torch.cuda.is_bf16_supported()
    if device.type == "cuda" and use_amp and not use_bfloat16:
        torch.set_float32_matmul_precision("high")
    print(f"Device: {device}, AMP: {use_amp}, BF16: {use_bfloat16}")

    # Tokenizer and vocab
    tokenizer = LatexTokenizer(vocab_path=vocab_path)
    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_id

    # Resolve max_samples from stage if set
    effective_max_samples = max_samples
    if stage is not None:
        effective_max_samples = STAGE_MAX_SAMPLES.get(stage, effective_max_samples)

    # WebDataset shards required
    shard_path = Path(shards_dir)
    if not shard_path.exists():
        raise RuntimeError(f"Shards directory not found: {shards_dir}. Run create_shards.py first.")
    shard_files = sorted(shard_path.glob("*.tar"))
    if not shard_files:
        raise RuntimeError("No WebDataset shards found in data_ready/shards. Run create_shards.py first.")
    shard_paths = [str(p) for p in shard_files]
    print(f"Using WebDataset: {len(shard_paths)} shards, max_samples={effective_max_samples}")

    dataset = WebFormulaDataset(
        shard_paths,
        tokenizer=tokenizer,
        img_height=128,
        width_multiple=32,
        max_len=max_len,
        add_sos_eos=True,
    )
    iterable_ds = dataset.iterable_dataset(max_samples=effective_max_samples)
    loader_kw = dict(
        dataset=iterable_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=partial(collate_fn_pad, pad_id=pad_id),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        loader_kw["prefetch_factor"] = 4
    loader = DataLoader(**loader_kw)

    # Model and optimizer
    model = MobileOneStudent(vocab_size=vocab_size, pad_id=pad_id).to(device)
    try:
        model = torch.compile(model)
    except Exception:
        pass
    if freeze_backbone:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("Encoder backbone frozen (use --unfreeze-epoch to unfreeze later)")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
        betas=(0.9, 0.98),
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_id,
        label_smoothing=0.1,
    )
    try:
        total_steps = max_epochs * len(loader)
    except TypeError:
        total_steps = max_epochs * 5000
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
    )
    scaler = GradScaler() if (use_amp and device.type == "cuda") else None

    # Optional teacher for knowledge distillation
    teacher = None
    if teacher_checkpoint is not None and Path(teacher_checkpoint).exists():
        teacher = MobileOneStudent(vocab_size=vocab_size, pad_id=pad_id).to(device)
        ckpt = torch.load(teacher_checkpoint, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        teacher.load_state_dict(state)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"Teacher loaded from {teacher_checkpoint}, distill_alpha={distill_alpha}, distill_temp={distill_temp}")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(1, max_epochs + 1):
        if freeze_backbone and unfreeze_epoch is not None and epoch == unfreeze_epoch:
            for p in model.encoder.parameters():
                p.requires_grad = True
            print(f"Epoch {epoch}: encoder unfrozen")

        epoch_max_len = max_len
        if curriculum:
            epoch_max_len = 64 if epoch < 3 else (96 if epoch < 8 else 128)

        running_loss = 0.0
        num_batches = 0
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if curriculum and epoch_max_len < labels.size(1):
                labels = labels[:, : epoch_max_len + 1]
            tgt_input = labels[:, :-1]
            tgt_output = labels[:, 1:]

            optimizer.zero_grad(set_to_none=True)
            if use_amp and device.type == "cuda":
                with autocast(dtype=torch.bfloat16 if use_bfloat16 else torch.float16):
                    logits = model(images, tgt_input)
                    seq_len = min(logits.size(1), tgt_output.size(1))
                    logits_flat = logits[:, :seq_len].reshape(-1, vocab_size)
                    labels_flat = tgt_output[:, :seq_len].reshape(-1)
                    ce_loss = criterion(logits_flat, labels_flat)
                    if teacher is not None:
                        with torch.no_grad():
                            teacher_logits = teacher(images, tgt_input)
                        teacher_logits_flat = teacher_logits[:, :seq_len].reshape(-1, vocab_size)
                        T_val = distill_temp
                        student_log_probs = F.log_softmax(logits_flat / T_val, dim=-1)
                        teacher_probs = F.softmax(teacher_logits_flat / T_val, dim=-1)
                        kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T_val * T_val)
                        loss = distill_alpha * ce_loss + (1.0 - distill_alpha) * kd_loss
                    else:
                        loss = ce_loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            else:
                logits = model(images, tgt_input)
                seq_len = min(logits.size(1), tgt_output.size(1))
                logits_flat = logits[:, :seq_len].reshape(-1, vocab_size)
                labels_flat = tgt_output[:, :seq_len].reshape(-1)
                ce_loss = criterion(logits_flat, labels_flat)
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits = teacher(images, tgt_input)
                    teacher_logits_flat = teacher_logits[:, :seq_len].reshape(-1, vocab_size)
                    T_val = distill_temp
                    student_log_probs = F.log_softmax(logits_flat / T_val, dim=-1)
                    teacher_probs = F.softmax(teacher_logits_flat / T_val, dim=-1)
                    kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T_val * T_val)
                    loss = distill_alpha * ce_loss + (1.0 - distill_alpha) * kd_loss
                else:
                    loss = ce_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            running_loss += loss.item()
            num_batches += 1

        mean_loss = running_loss / max(1, num_batches)
        print(f"Epoch {epoch}/{max_epochs} Loss: {mean_loss:.4f}")

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
    p = argparse.ArgumentParser(description="Screen2LaTeX production training (WebDataset shards only)")
    p.add_argument("--data-dir", type=str, default="./data_ready", help="Data dir (contains shards/, vocab.txt)")
    p.add_argument("--vocab", type=str, default=None, help="Path to vocab.txt (default: data-dir/vocab.txt)")
    p.add_argument("--shards-dir", type=str, default=None, help="WebDataset shards dir (default: data-dir/shards)")
    p.add_argument("--save-dir", type=str, default="checkpoints", help="Checkpoint directory")
    p.add_argument("--batch-size", type=int, default=96, help="Batch size")
    p.add_argument("--epochs", type=int, default=100, help="Max epochs")
    p.add_argument("--max-len", type=int, default=128, help="Max sequence length")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--checkpoint-every", type=int, default=5, help="Save checkpoint every N epochs")
    p.add_argument("--no-amp", action="store_true", help="Disable AMP")
    p.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    p.add_argument("--teacher-checkpoint", type=str, default=None, help="Teacher checkpoint for knowledge distillation")
    p.add_argument("--distill-alpha", type=float, default=0.5, help="Weight for CE loss in distillation")
    p.add_argument("--distill-temp", type=float, default=2.0, help="Temperature for distillation softmax")
    p.add_argument("--max-samples", type=int, default=None, help="Limit samples per epoch (WebDataset with_epoch)")
    p.add_argument("--stage", type=str, default=None, choices=("stage1", "stage2", "stage3", "stage4"),
                   help="Progressive scaling: stage1=50k, stage2=200k, stage3=500k, stage4=full")
    p.add_argument("--freeze-backbone", action="store_true", help="Freeze MobileOne encoder at start")
    p.add_argument("--unfreeze-epoch", type=int, default=None, help="Epoch at which to unfreeze encoder")
    p.add_argument("--curriculum", action="store_true", help="Sequence length curriculum: 64->96->128 by epoch")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    vocab_path = args.vocab or str(data_dir / "vocab.txt")
    shards_dir = args.shards_dir or str(data_dir / "shards")

    train(
        vocab_path=vocab_path,
        save_dir=args.save_dir,
        shards_dir=shards_dir,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        max_len=args.max_len,
        lr=args.lr,
        checkpoint_every=args.checkpoint_every,
        use_amp=not args.no_amp,
        num_workers=args.num_workers,
        teacher_checkpoint=args.teacher_checkpoint,
        distill_alpha=args.distill_alpha,
        distill_temp=args.distill_temp,
        max_samples=args.max_samples,
        stage=args.stage,
        freeze_backbone=args.freeze_backbone,
        unfreeze_epoch=args.unfreeze_epoch,
        curriculum=args.curriculum,
    )


if __name__ == "__main__":
    main()
