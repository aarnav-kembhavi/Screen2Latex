"""
LatexTokenizer: regex-based tokenizer for LaTeX formulas (UniMER / Screen2LaTeX).
"""

import re
from pathlib import Path
from typing import List, Optional

# Special tokens (order must match typical vocab indexing)
PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIAL_TOKENS = [PAD, SOS, EOS, UNK]

# Regex: letters, dot, braces/underscore/caret, or single non-space character
TOKEN_PATTERN = re.compile(r"[a-zA-Z]+|\.|[{}_^]|[^ \t\n]")


class LatexTokenizer:
    """
    Tokenizer for LaTeX formulas using regex.
    Special tokens: <pad>, <sos>, <eos>, <unk>.
    """

    def __init__(self, vocab_path: Optional[str] = None, token2id: Optional[dict] = None):
        """
        Args:
            vocab_path: Path to vocab.txt (one token per line).
            token2id: Optional pre-built token -> id map (overrides vocab_path if both given).
        """
        self.token2id: dict = {}
        self.id2token: dict = {}
        if token2id is not None:
            self.token2id = dict(token2id)
            self.id2token = {i: t for t, i in self.token2id.items()}
        elif vocab_path is not None:
            self._load_vocab(vocab_path)
        else:
            # Minimal vocab: only special tokens
            for i, t in enumerate(SPECIAL_TOKENS):
                self.token2id[t] = i
                self.id2token[i] = t

    def _load_vocab(self, path: str) -> None:
        """Load token2id from vocab file (one token per line)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Vocab file not found: {path}")
        tokens = []
        with open(path, "r", encoding="utf-8") as f:
            tokens = [line.strip() for line in f if line.strip()]
        for t in SPECIAL_TOKENS:
            if t not in tokens:
                tokens.insert(0, t)
        for i, t in enumerate(tokens):
            self.token2id[t] = i
            self.id2token[i] = t

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    @property
    def pad_id(self) -> int:
        return self.token2id[PAD]

    @property
    def sos_id(self) -> int:
        return self.token2id[SOS]

    @property
    def eos_id(self) -> int:
        return self.token2id[EOS]

    @property
    def unk_id(self) -> int:
        return self.token2id[UNK]

    def tokenize(self, formula: str) -> List[str]:
        """Split formula into list of tokens (regex)."""
        return TOKEN_PATTERN.findall(formula)

    def encode(self, formula: str, max_len: Optional[int] = None, add_sos_eos: bool = True) -> List[int]:
        """
        Encode formula to list of token ids.
        If add_sos_eos: [sos] + token_ids + [eos]. Padded with pad_id to max_len.
        """
        tokens = self.tokenize(formula)
        ids = [self.token2id.get(t, self.unk_id) for t in tokens]
        if add_sos_eos:
            ids = [self.sos_id] + ids + [self.eos_id]
        if max_len is not None:
            if add_sos_eos:
                max_tokens = max_len - 2
                ids = ids[1:-1][:max_tokens]
                ids = [self.sos_id] + ids + [self.eos_id]
            else:
                ids = ids[:max_len]
            if len(ids) < max_len:
                ids = ids + [self.pad_id] * (max_len - len(ids))
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Decode list of token ids to string.
        If skip_special: strip <pad>, <sos>, <eos>, <unk> from output.
        """
        tokens = []
        for i in ids:
            t = self.id2token.get(i, UNK)
            if skip_special and t in {PAD, SOS, EOS}:
                if t == EOS:
                    break
                continue
            tokens.append(t)
        return "".join(tokens)

    @staticmethod
    def build_vocab(formulas: List[str], special_first: bool = True) -> dict:
        """
        Build token2id from list of formulas.
        Collects all unique tokens from regex split; special tokens first.
        Returns token2id dict.
        """
        token2id = {t: i for i, t in enumerate(SPECIAL_TOKENS)} if special_first else {}
        for formula in formulas:
            for t in TOKEN_PATTERN.findall(formula):
                if t not in token2id:
                    token2id[t] = len(token2id)
        return token2id

    def save_vocab(self, path: str) -> None:
        """Save vocab to file (one token per line, order by id)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for i in range(len(self.id2token)):
                f.write(self.id2token[i] + "\n")
