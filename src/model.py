"""
MobileOneStudent: CNN (MobileOne-S0) encoder + Bi-GRU decoder for LaTeX formula recognition.
Production: reparameterize() for NCNN / inference.
"""

try:
    import torch
    import torch.nn as nn
except ImportError as e:
    raise ImportError("PyTorch is required. Install with: pip install torch") from e

try:
    import timm
    from timm.utils.model import reparameterize_model
except ImportError as e:
    raise ImportError("timm is required. Install with: pip install timm") from e


class MobileOneStudent(nn.Module):
    """
    Screen2LaTeX student: MobileOne-S0 backbone (last feature map 1024ch @ stride 32)
    -> 1x1 projector (256ch) -> vertical average pooling -> 2-layer Bi-GRU -> linear head.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

        # ----- Encoder: timm MobileOne-S0, features_only => list of feature maps -----
        # Last map: [B, 1024, H, W] (stride 32; H,W depend on input size)
        self.encoder = timm.create_model(
            "mobileone_s0",
            pretrained=True,
            features_only=True,
        )

        # ----- Neck/Projector: reduce channels for RNN -----
        # 1024 (MobileOne-S0 last stage) -> 256
        self.projector = nn.Conv2d(1024, 256, kernel_size=1)

        # ----- Pooling: vertical-only (mean over height); handled in forward -----

        # ----- Decoder: 2-layer Bi-GRU (CRITICAL) -----
        # input 256, hidden 256, 2 layers, bidirectional => output 512 per step
        self.decoder = nn.GRU(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        # ----- Head: 512 (256*2 from Bi-GRU) -> vocab -----
        self.head = nn.Linear(512, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H_in, W_in]  (e.g. [B, 3, 128, W])
        Returns:
            logits: [B, Seq_Len, vocab_size]
        """
        # ----- CNN: encoder (features_only returns list; last = highest stride) -----
        features = self.encoder(x)
        # features: list; last element shape [B, 1024, H, W]
        cnn_out = features[-1]
        # cnn_out shape: [B, 1024, H, W]

        # ----- Projector: 1x1 conv -----
        proj = self.projector(cnn_out)
        # proj shape: [B, 256, H, W]

        # ----- Vertical average pooling: collapse Height (dim=2) to 1 -----
        # mean(dim=2) => [B, 256, W]; each width column = one time step
        pooled = proj.mean(dim=2)
        # pooled shape: [B, 256, W]  (W = sequence length)

        # ----- Permute to [Batch, Seq_Len, 256] for batch_first=True GRU -----
        seq = pooled.permute(0, 2, 1)
        # seq shape: [B, Seq_Len, 256]

        # ----- Bi-GRU -----
        rnn_out, _ = self.decoder(seq)
        # rnn_out shape: [B, Seq_Len, 512]

        # ----- Head: per-step logits -----
        logits = self.head(rnn_out)
        # logits shape: [B, Seq_Len, vocab_size]

        return logits

    def reparameterize(self) -> None:
        """
        Collapse MobileOne branches for NCNN / inference.
        Calls timm reparameterize on the encoder (in-place).
        """
        reparameterize_model(self.encoder, inplace=True)
