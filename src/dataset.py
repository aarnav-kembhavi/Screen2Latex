"""
WebDataset pipeline for Screen2LaTeX: tar shards (jpg + txt) -> decode, resize, tokenize.
UniMER-1M compatible. No legacy train.txt or image-dir support.
"""

from typing import Callable, Optional, Tuple

try:
    import torch
    from torch.utils.data import IterableDataset
except ImportError as e:
    raise ImportError("PyTorch is required. Install with: pip install torch") from e

try:
    from torchvision import transforms
    from PIL import Image
except ImportError as e:
    raise ImportError("torchvision, Pillow required.") from e

from src.tokenizer import LatexTokenizer

try:
    import webdataset as wds
except ImportError:
    wds = None


def _to_rgb(img: Image.Image) -> Image.Image:
    """Convert grayscale to RGB (3 channels for MobileOne)."""
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def pad_width_to_multiple(w: int, multiple: int = 32) -> int:
    """Return smallest width >= w that is multiple of `multiple`."""
    return ((w + multiple - 1) // multiple) * multiple


def _webdataset_preprocess(
    sample,
    tokenizer: LatexTokenizer,
    img_height: int,
    width_multiple: int,
    max_len: Optional[int],
    add_sos_eos: bool,
    to_tensor: Callable,
    normalize: Callable,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map (PIL image, formula bytes) -> (image_tensor, label_tensor) for WebDataset."""
    image, formula_bytes = sample[0], sample[1]
    formula = formula_bytes.decode("utf-8") if isinstance(formula_bytes, bytes) else formula_bytes
    image = _to_rgb(image)
    w, h = image.size
    new_w = max(1, int(w * (img_height / h)))
    new_w = min(new_w, 1024)  # cap before multiple rounding
    new_w = pad_width_to_multiple(new_w, width_multiple)
    image = image.resize((new_w, img_height), Image.Resampling.BILINEAR)
    image_tensor = normalize(to_tensor(image))
    ids = tokenizer.encode(formula, max_len=max_len, add_sos_eos=add_sos_eos)
    label_tensor = torch.tensor(ids, dtype=torch.long)
    return (image_tensor, label_tensor)

class WebFormulaDataset:
    """
    WebDataset pipeline for Screen2LaTeX: tar shards -> decode images -> resize -> tokenize -> (image, label).
    Use with DataLoader; collate_fn_pad for batching. Install: pip install webdataset
    """

    def __init__(
        self,
        shards,
        tokenizer: LatexTokenizer,
        img_height: int = 128,
        width_multiple: int = 32,
        max_len: Optional[int] = 128,
        add_sos_eos: bool = True,
    ):
        if wds is None:
            raise ImportError("WebDataset required for WebFormulaDataset. pip install webdataset")
        self.shards = shards
        self.tokenizer = tokenizer
        self.img_height = img_height
        self.width_multiple = width_multiple
        self.max_len = max_len
        self.add_sos_eos = add_sos_eos
        self._to_tensor = transforms.ToTensor()
        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def _build_pipeline(self, max_samples: Optional[int] = None):
        def preprocess(sample):
            return _webdataset_preprocess(
                sample,
                self.tokenizer,
                self.img_height,
                self.width_multiple,
                self.max_len,
                self.add_sos_eos,
                self._to_tensor,
                self._normalize,
            )

        dataset = (
            wds.WebDataset(self.shards, resampled=False)
            .shuffle(2000, initial=2000)
            .decode("pil")
            .to_tuple("jpg", "txt")
            .map(preprocess, handler=wds.handlers.warn_and_continue)
        )
        if max_samples is not None:
            dataset = dataset.with_epoch(max_samples)
        return dataset

    def pipeline(self, max_samples: Optional[int] = None):
        """Return pipeline (single-use iterator). For multi-epoch training use iterable_dataset()."""
        return self._build_pipeline(max_samples)

    def iterable_dataset(self, max_samples: Optional[int] = None) -> "WebFormulaIterableDataset":
        """Return an IterableDataset that creates a fresh pipeline each epoch (for DataLoader)."""
        return WebFormulaIterableDataset(self, max_samples)


class WebFormulaIterableDataset(IterableDataset):
    """Wraps WebFormulaDataset so each epoch gets a fresh pipeline (DataLoader calls __iter__ per epoch)."""

    def __init__(self, wds_dataset: WebFormulaDataset, max_samples: Optional[int] = None):
        super().__init__()
        self._wds_dataset = wds_dataset
        self._max_samples = max_samples

    def __iter__(self):
        return iter(self._wds_dataset._build_pipeline(self._max_samples))


def collate_fn_pad(batch: list, pad_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate batch of (image, label) with variable width images and variable label length.
    Pads images to max width in batch (already height 128 and width multiple of 32 per sample).
    Pads labels to max length in batch.
    Returns: images [B, 3, H, W_max], labels [B, L_max], label_lengths [B] (for loss masking if needed).
    """
    images = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    max_w = max(im.size(2) for im in images)
    max_l = max(lab.size(0) for lab in labels)
    label_lengths = [lab.size(0) for lab in labels]
    # Pad images to max_w (right pad with 0)
    h = images[0].size(1)
    padded_images = []
    for im in images:
        w = im.size(2)
        if w < max_w:
            pad = torch.zeros(im.size(0), h, max_w - w, dtype=im.dtype)
            im = torch.cat([im, pad], dim=2)
        padded_images.append(im)
    # Pad labels to max_l with pad_id
    padded_labels = []
    for lab in labels:
        if lab.size(0) < max_l:
            lab = torch.cat([lab, torch.full((max_l - lab.size(0),), pad_id, dtype=lab.dtype)])
        padded_labels.append(lab)
    return torch.stack(padded_images), torch.stack(padded_labels), torch.tensor(label_lengths)
