from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class Observation:
    images: dict[str, Any]
    image_masks: dict[str, Any]
    state: Any
    tokenized_prompt: Any | None = None
    tokenized_prompt_mask: Any | None = None
    token_ar_mask: Any | None = None
    token_loss_mask: Any | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "Observation":
        images = data["image"]
        for key, image in list(images.items()):
            if isinstance(image, torch.Tensor) and image.dtype == torch.uint8:
                images[key] = image.to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
            elif isinstance(image, np.ndarray) and image.dtype == np.uint8:
                images[key] = image.astype(np.float32) / 255.0 * 2.0 - 1.0
        return cls(
            images=images,
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )

    def to_dict(self) -> dict:
        return {
            "image": self.images,
            "image_mask": self.image_masks,
            "state": self.state,
            "tokenized_prompt": self.tokenized_prompt,
            "tokenized_prompt_mask": self.tokenized_prompt_mask,
            "token_ar_mask": self.token_ar_mask,
            "token_loss_mask": self.token_loss_mask,
        }

