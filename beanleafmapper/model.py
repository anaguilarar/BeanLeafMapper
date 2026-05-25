"""SAM3 model loader and a thin wrapper around its image processor."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image


@dataclass
class Sam3Inference:
    """SAM3 inference state from a single image + text prompt."""

    scores: torch.Tensor
    boxes: torch.Tensor
    masks: torch.Tensor
    raw: dict[str, Any]

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "Sam3Inference":
        return cls(
            scores=state["scores"],
            boxes=state["boxes"],
            masks=state["masks"],
            raw=state,
        )


def configure_torch() -> None:
    """Enable TF32 + autocast on supported GPUs.

    bfloat16 needs compute capability >= 8.0 (Ampere+); on older GPUs we fall back
    to float16, which is supported from Volta onward."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not torch.cuda.is_available():
        return
    major, _ = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if major >= 8 else torch.float16
    torch.autocast("cuda", dtype=dtype).__enter__()


def login_huggingface(token: str | None = None) -> None:
    """Authenticate with Hugging Face. Reads HF_TOKEN env if no token given."""
    from huggingface_hub import login

    tok = token or os.environ.get("HF_TOKEN")
    if tok:
        login(tok)


class Sam3Model:
    """Wraps the SAM3 image model + processor for text-prompted segmentation."""

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        checkpoint_path: str | None = None,
        device: str | None = None,
        compile: bool = False,
    ):
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        configure_torch()
        build_kwargs: dict[str, object] = {"compile": compile}
        if checkpoint_path is not None:
            build_kwargs["checkpoint_path"] = checkpoint_path
        if device is not None:
            build_kwargs["device"] = device
        self._model = build_sam3_image_model(**build_kwargs)
        self._processor_cls = Sam3Processor
        self._confidence_threshold = confidence_threshold

    def detect(
        self,
        image: Image.Image,
        text_prompt: str,
        confidence_threshold: float | None = None,
    ) -> Sam3Inference:
        """Run SAM3 detection with a text prompt and return the inference state."""
        threshold = confidence_threshold or self._confidence_threshold
        processor = self._processor_cls(self._model, confidence_threshold=threshold)
        state = processor.set_image(image)
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(state=state, prompt=text_prompt)
        return Sam3Inference.from_state(state)


def build_detector(cfg: "ModelConfig", confidence_threshold: float) -> "Sam3Model":
    """Construct the configured text-promptable detector.

    Today only `backend="sam3_image"` is implemented; other backends are listed in
    README and will raise NotImplementedError until wired in."""
    from .config import ModelConfig as _MC  # noqa: F401  (typing only)

    if cfg.backend == "sam3_image":
        return Sam3Model(
            confidence_threshold=confidence_threshold,
            checkpoint_path=cfg.checkpoint_path,
            device=cfg.device,
            compile=cfg.compile,
        )
    raise NotImplementedError(
        f"Model backend {cfg.backend!r} is not implemented. "
        f"Implemented: 'sam3_image'. See README for the menu of planned backends."
    )
