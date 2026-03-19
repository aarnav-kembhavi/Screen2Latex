"""
WebDataset pipeline for Screen2LaTeX: tar shards (npy + txt) -> load, tokenize.
Images are pre-decoded and pre-resized numpy arrays — zero CPU decode at training time.
"""

from typing import Optional, Tuple

try:
    import torch
    from torch.utils.data import IterableDataset
except ImportError as e:
    raise ImportError("PyTorch is required. Install with: pip install torch") from e

try:
    from torchvision import transforms
except ImportError as e:
    raise ImportError("torchvision required.") from e

import numpy as np
from src.tokenizer import LatexTokenizer

try:
    import webdataset as wds
except ImportError:
    wds = None

# ImageNet normalization (same as before)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def pad_width_to_multiple(w: int, multiple: int = 32) -> int:
    return ((w + multiple - 1) // multiple) * multiple


def _preprocess(
    sample,
    tokenizer: LatexTokenizer,
    max_len: Optional[int],
    add_sos_eos: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load pre-processed numpy image + tokenize label. No PIL, no resize."""
    npy_bytes, formula_bytes = sample[0], sample[1]

    # Load numpy array: shape (H, W, 3), dtype uint8
    arr = np.frombuffer(npy_bytes, dtype=np.uint8).copy()
    img = torch.from_numpy(
        np.load(__import__("io").BytesIO(npy_bytes))
    ).permute(2, 0, 1).float().div(255.0)  # [3, H, W] float32

    # Normalize
    img = (img - _MEAN) / _STD

    # Tokenize
    formula = formula_bytes.decode("utf-8") if isinstance(formula_bytes, bytes) else formula_bytes
    ids = tokenizer.encode(formula, max_len=max_len, add_sos_eos=add_sos_eos)
    label = torch.tensor(ids, dtype=torch.long)

    return (img, label)


class WebFormulaDataset:
    """
    WebDataset pipeline: tar shards (npy + txt) -> load tensor -> tokenize.
    Shards must be created with the numpy-aware create_shards.py.
    """

    def __init__(
        self,
        shards,
        tokenizer: LatexTokenizer,
        img_height: int = 128,        # kept for API compatibility, not used at load time
        width_multiple: int = 32,     # kept for API compatibility
        max_len: Optional[int] = 128,
        add_sos_eos: bool = True,
    ):
        if wds is None:
            raise ImportError("WebDataset required. pip install webdataset")
        self.shards = shards
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.add_sos_eos = add_sos_eos

    def _build_pipeline(self, max_samples: Optional[int] = None):
        def preprocess(sample):
            return _preprocess(
                sample,
                self.tokenizer,
                self.max_len,
                self.add_sos_eos,
            )

        dataset = (
            wds.WebDataset(self.shards, resampled=False, shardshuffle=True)
            .shuffle(500, initial=500)
            .to_tuple("npy", "txt")
            .map(preprocess, handler=wds.handlers.warn_and_continue)
        )
        if max_samples is not None:
            dataset = dataset.with_epoch(max_samples)
        return dataset

    def pipeline(self, max_samples: Optional[int] = None):
        return self._build_pipeline(max_samples)

    def iterable_dataset(self, max_samples: Optional[int] = None) -> "WebFormulaIterableDataset":
        return WebFormulaIterableDataset(self, max_samples)


class WebFormulaIterableDataset(IterableDataset):
    def __init__(self, wds_dataset: WebFormulaDataset, max_samples: Optional[int] = None):
        super().__init__()
        self._wds_dataset = wds_dataset
        self._max_samples = max_samples

    def __iter__(self):
        return iter(self._wds_dataset._build_pipeline(self._max_samples))


def collate_fn_pad(batch: list, pad_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad images to max width in batch, pad labels to max length.
    Returns: images [B, 3, H, W_max], labels [B, L_max], lengths [B]
    """
    images = [b[0] for b in batch]
    labels = [b[1] for b in batch]

    max_w = max(im.size(2) for im in images)
    max_l = max(lab.size(0) for lab in labels)
    label_lengths = [lab.size(0) for lab in labels]
    h = images[0].size(1)

    padded_images = []
    for im in images:
        w = im.size(2)
        if w < max_w:
            pad = torch.zeros(im.size(0), h, max_w - w, dtype=im.dtype)
            im = torch.cat([im, pad], dim=2)
        padded_images.append(im)

    padded_labels = []
    for lab in labels:
        if lab.size(0) < max_l:
            lab = torch.cat([lab, torch.full((max_l - lab.size(0),), pad_id, dtype=lab.dtype)])
        padded_labels.append(lab)

    return torch.stack(padded_images), torch.stack(padded_labels), torch.tensor(label_lengths)