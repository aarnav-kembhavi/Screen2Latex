"""
MobileOneStudent: Autoregressive transformer encoder-decoder for LaTeX OCR with teacher forcing.
Lightweight (~6.5M params). reparameterize() for NCNN / inference.
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
    Screen2LaTeX: MobileOne encoder -> 1x1 projection -> flatten tokens
    -> pos encoding -> Transformer Encoder (4L) -> token embedding + decoder pos
    -> Transformer Decoder (2L, causal + cross-attn) -> linear head. Autoregressive with teacher forcing.
    """

    def __init__(self, vocab_size: int, pad_id: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id

        # ----- Encoder: timm MobileOne-S0, features_only => list of feature maps -----
        self.encoder = timm.create_model(
            "mobileone_s0",
            pretrained=True,
            features_only=True,
        )

        # ----- 1x1 projection -----
        self.projector = nn.Conv2d(1024, 256, 1)

        # ----- 2D spatial positional embeddings (row + column); 1024 cols for long equations -----
        self.row_embed = nn.Embedding(64, 256)
        self.col_embed = nn.Embedding(1024, 256)

        # ----- Transformer Encoder (4 layers) -----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=512,
            activation="gelu",
            batch_first=True,
        )
        self.encoder_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4,
        )

        # ----- Decoder: token embedding + learned positional embedding -----
        self.max_seq_len = 128
        self.token_embed = nn.Embedding(vocab_size, 256)
        self.decoder_pos = nn.Embedding(self.max_seq_len, 256)

        # ----- Transformer Decoder (2 layers, cross attention to encoder output) -----
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=512,
            activation="gelu",
            batch_first=True,
        )
        self.decoder_transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=2,
        )
        self.decoder = self.decoder_transformer  # alias for hooks / compatibility

        # ----- Output head -----
        self.head = nn.Linear(256, vocab_size)

    def generate_square_subsequent_mask(self, sz, device):
        """Causal mask: prevent attending to future tokens."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to memory. Use this once at inference and pass memory to decode().
        Args:
            x: [B, C, H_in, W_in]  (e.g. [B, 3, 128, W])
        Returns:
            memory: [B, H*W, 256]
        """
        features = self.encoder(x)
        cnn_out = features[-1]  # [B, 1024, H, W]
        proj = self.projector(cnn_out)  # [B, 256, H, W]

        B, C, H, W = proj.shape
        rows = torch.arange(H, device=proj.device).clamp(max=63)
        cols = torch.arange(W, device=proj.device).clamp(max=1023)
        row_emb = self.row_embed(rows)[:, None, :]   # H x 1 x C
        col_emb = self.col_embed(cols)[None, :, :]   # 1 x W x C
        pos2d = row_emb + col_emb                    # H x W x C
        pos2d = pos2d.permute(2, 0, 1).unsqueeze(0)
        proj = proj + pos2d

        tokens = proj.flatten(2).permute(0, 2, 1)  # [B, H*W, 256]
        memory = self.encoder_transformer(tokens)  # [B, H*W, 256]
        return memory

    def decode(self, memory: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        """
        Autoregressive decode from cached encoder memory.
        Args:
            memory: [B, H*W, 256] from encode()
            tgt_tokens: [B, seq_len] decoder input token ids
        Returns:
            logits: [B, seq_len, vocab_size]
        """
        tgt = self.token_embed(tgt_tokens)  # [B, seq_len, 256]
        pos = torch.arange(tgt.size(1), device=tgt.device)
        tgt = tgt + self.decoder_pos(pos).unsqueeze(0)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), tgt.device)
        tgt_padding_mask = tgt_tokens == self.pad_id
        decoder_out = self.decoder_transformer(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )  # [B, seq_len, 256]
        logits = self.head(decoder_out)  # [B, seq_len, vocab_size]
        return logits

    def forward(self, x: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H_in, W_in]  (e.g. [B, 3, 128, W])
            tgt_tokens: [B, seq_len]  (decoder input token ids; e.g. ground truth for teacher forcing)
        Returns:
            logits: [B, seq_len, vocab_size]
        """
        memory = self.encode(x)
        return self.decode(memory, tgt_tokens)

    def reparameterize(self) -> None:
        """
        Collapse MobileOne branches for NCNN / inference.
        """
        reparameterize_model(self.encoder, inplace=True)
