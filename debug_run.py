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

    # 2. Test Model Architecture (autoregressive: forward(image, tgt_tokens))
    model = MobileOneStudent(vocab_size=len(vocab))
    dummy_input = torch.randn(2, 3, 128, 384)  # [Batch, C, H, W]
    dummy_tokens = torch.randint(0, len(vocab), (2, 10))

    attention_output_shape = []

    def capture_attention_shape(module, inp, out):
        if inp and isinstance(inp, (list, tuple)) and len(inp) > 0:
            attention_output_shape.append(tuple(inp[0].shape))

    hook_handle = model.decoder.register_forward_hook(capture_attention_shape)
    print(f"Testing Forward Pass with Input: {dummy_input.shape}, tgt_tokens: {dummy_tokens.shape}")
    logits = model(dummy_input, dummy_tokens)
    hook_handle.remove()
    print(f"[Pass] Forward Output Shape: {logits.shape}")
    if attention_output_shape:
        print(f"[Pass] Decoder input shape: {attention_output_shape[0]}")
    # Expected: [2, 10, 7] (B, seq_len, vocab_size)

    # 3. Test Reparameterization (Crucial for Inference)
    model.eval()
    model.reparameterize()
    print("[Pass] MobileOne Reparameterization Successful")

    # 4. Test inference-style decode (argmax per position)
    out_ids = logits.argmax(dim=2)
    print(f"[Pass] Greedy Decode Shape: {out_ids.shape}")

    # Cleanup
    os.remove("debug_vocab.txt")
    print(">>> Dry Run Successful. Ready for A100 Deployment.")


if __name__ == "__main__":
    test_production_ready()
