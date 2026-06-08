from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import sentencepiece


def default_tokenizer_path() -> Path:
    package_path = Path(__file__).resolve().parent / "assets" / "paligemma_tokenizer.model"
    candidates = [
        os.environ.get("PALIGEMMA_TOKENIZER_PATH"),
        str(package_path) if package_path.exists() else None,
        str(Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError(
        "PaliGemma tokenizer model not found. Set PALIGEMMA_TOKENIZER_PATH or place "
        "paligemma_tokenizer.model at src/datamil_pi0/assets/paligemma_tokenizer.model."
    )


class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48, tokenizer_path: str | os.PathLike | None = None):
        self._max_len = max_len
        path = Path(tokenizer_path) if tokenizer_path is not None else default_tokenizer_path()
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")

        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if tokens_len > self._max_len:
                logging.warning("Token length %s exceeds max length %s; truncating.", tokens_len, self._max_len)
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len
        return np.asarray(tokens, dtype=np.int32), np.asarray(mask, dtype=bool)

