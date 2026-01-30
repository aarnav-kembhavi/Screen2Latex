import torch
import torchvision.transforms as T
from PIL import Image
from .model import MobileOneStudent
from .tokenizer import LatexTokenizer


class LatexPredictor:
    def __init__(self, checkpoint_path, vocab_file, device="cpu"):
        self.device = device
        self.tokenizer = LatexTokenizer(vocab_path=vocab_file)

        # Load Model
        self.model = MobileOneStudent(vocab_size=self.tokenizer.vocab_size)

        # Load Weights
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle case where checkpoint saves 'model_state_dict' key
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)

        # Optimize for Inference
        self.model.to(device)
        self.model.eval()
        if hasattr(self.model, "reparameterize"):
            self.model.reparameterize()  # Critical for MobileOne Speed

        # Preprocessing
        self.transform = T.Compose([
            T.Grayscale(num_output_channels=3),
            T.Resize((128, 384)),  # Use fixed size for simple testing
            T.ToTensor(),
        ])

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Forward Pass
            logits = self.model(img_tensor)  # [1, Seq, Vocab]

            # Greedy Decode (Argmax)
            # This assumes CTC-like behavior where we pick the best token per frame
            preds = logits.argmax(dim=2)  # [1, Seq]

            # Convert IDs to Tokens
            tokens = []
            prev_token = None
            for idx in preds[0]:
                token_id = idx.item()
                token = self.tokenizer.id2token.get(token_id, "")

                # Simple logic: skip pads, specials, and duplicates (if desired)
                if token in ["<pad>", "<sos>", "<eos>", "<unk>"]:
                    continue
                # Optional: Dedup for CTC (only add if different from prev)
                # if token_id != prev_token:
                tokens.append(token)
                prev_token = token_id

            return " ".join(tokens)


# CLI usage
if __name__ == "__main__":
    import sys

    # Example usage: python src/inference.py image.png
    # Requires checkpoints/best_model.pt and vocab.txt to exist
    predictor = LatexPredictor("checkpoints/best_model.pt", "vocab.txt")
    print(predictor.predict(sys.argv[1]))
