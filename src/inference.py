import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from .model import MobileOneStudent
from .tokenizer import LatexTokenizer

# Length penalty exponent for beam scoring
BEAM_LENGTH_PENALTY_ALPHA = 0.6


def beam_search_decode(model, img_tensor, tokenizer, beam_size=3, max_len=128):
    """
    Beam search decoding: encode image once, then decode autoregressively.
    Only expands top-k tokens per step (no full vocabulary). Beams sorted by
    score / (len(sequence) ** alpha).
    """
    device = img_tensor.device
    sos_id = tokenizer.sos_id
    eos_id = tokenizer.eos_id
    alpha = BEAM_LENGTH_PENALTY_ALPHA

    with torch.inference_mode():
        memory = model.encode(img_tensor)  # [1, H*W, 256]; compute once

    beams = [([sos_id], 0.0)]  # (sequence, cumulative log prob)
    for _ in range(max_len - 1):
        all_seqs = [b[0] for b in beams]
        batch_tokens = torch.tensor(all_seqs, dtype=torch.long, device=device)
        memory_expanded = memory.repeat(len(beams), 1, 1)
        with torch.inference_mode():
            logits = model.decode(memory_expanded, batch_tokens)
        log_probs = F.log_softmax(logits[:, -1], dim=-1)

        candidates = []
        for i, (seq, score) in enumerate(beams):
            if seq[-1] == eos_id:
                candidates.append((seq, score))
                continue
            topk = torch.topk(log_probs[i], beam_size)
            for token, log_prob in zip(topk.indices.tolist(), topk.values.tolist()):
                new_seq = seq + [token]
                new_score = score + log_prob
                candidates.append((new_seq, new_score))

        # Sort by length-normalized score: score / (len(seq) ** alpha)
        candidates.sort(key=lambda x: -(x[1] / ((len(x[0])) ** alpha)))
        beams = candidates[:beam_size]
        if all(b[0][-1] == eos_id for b in beams):
            break

    best = beams[0][0]
    return tokenizer.decode(best)


def resize_with_aspect(image, target_height=128, width_multiple=32):
    """Resize image preserving aspect ratio, then pad width to multiple (matches training)."""
    w, h = image.size
    new_h = target_height
    new_w = max(1, int(w * (new_h / h)))
    new_w = ((new_w + width_multiple - 1) // width_multiple) * width_multiple
    return image.resize((new_w, new_h), Image.Resampling.BILINEAR)


def _default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class LatexPredictor:
    def __init__(self, checkpoint_path, vocab_file, device=None):
        self.device = device if device is not None else _default_device()
        self.tokenizer = LatexTokenizer(vocab_path=vocab_file)

        self.model = MobileOneStudent(
            vocab_size=self.tokenizer.vocab_size,
            pad_id=self.tokenizer.pad_id,
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()
        if hasattr(self.model, "reparameterize"):
            self.model.reparameterize()

        self.transform = T.Compose([
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def predict(self, image_path, max_len=128, beam_size=3):
        image = Image.open(image_path).convert("RGB")
        image = resize_with_aspect(image, target_height=128, width_multiple=32)
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return beam_search_decode(
            self.model,
            img_tensor,
            self.tokenizer,
            beam_size=beam_size,
            max_len=max_len,
        )


if __name__ == "__main__":
    import sys
    predictor = LatexPredictor("checkpoints/best_model.pt", "vocab.txt")
    print(predictor.predict(sys.argv[1]))
