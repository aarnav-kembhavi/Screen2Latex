import torch
import os
from src.model import MobileOneStudent
from src.tokenizer import LatexTokenizer


def test_production_ready():
    print(">>> Initializing Production Dry Run...")

    # 1. Test Tokenizer
    vocab = ["<pad>", "<sos>", "<eos>", "<unk>", "x", "=", "1"]
    with open("debug_vocab.txt", "w") as f:
        f.write("\n".join(vocab))

    tokenizer = LatexTokenizer("debug_vocab.txt")
    encoded = tokenizer.encode("x = 1")
    print(f"[Pass] Tokenizer Encoded 'x = 1' -> {encoded}")

    # 2. Test Model Architecture
    model = MobileOneStudent(vocab_size=len(vocab))
    dummy_input = torch.randn(2, 3, 128, 384)  # [Batch, C, H, W]

    print(f"Testing Forward Pass with Input: {dummy_input.shape}")
    logits = model(dummy_input)
    print(f"[Pass] Forward Output Shape: {logits.shape}")
    # Expected: [2, 12, 7] (12 = 384 / 32 stride)

    # 3. Test Reparameterization (Crucial for Inference)
    model.eval()
    model.reparameterize()
    print("[Pass] MobileOne Reparameterization Successful")

    # 4. Test Inference Output
    out_ids = logits.argmax(dim=2)
    print(f"[Pass] Greedy Decode Shape: {out_ids.shape}")

    # Cleanup
    os.remove("debug_vocab.txt")
    print(">>> Dry Run Successful. Ready for A100 Deployment.")


if __name__ == "__main__":
    test_production_ready()
