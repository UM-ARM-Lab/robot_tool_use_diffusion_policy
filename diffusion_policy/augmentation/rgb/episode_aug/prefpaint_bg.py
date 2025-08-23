# episode_aug/prefpaint_bg.py
import os
from typing import Optional, Sequence, Tuple
import numpy as np
from PIL import Image
import torch

# PrefPaint (diffusers) import
from diffusers import AutoPipelineForInpainting

def _to_pil_rgb(img: np.ndarray) -> Image.Image:
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8
    return Image.fromarray(img)  # RGB

def _to_pil_mask(mask_bool: np.ndarray) -> Image.Image:
    """
    diffusers inpainting expects WHITE (255) where we want to INPAINT (remove/replace),
    and BLACK (0) where we want to KEEP.
    Here, we assume mask_bool == True is foreground to remove.
    """
    assert mask_bool.ndim == 2
    m = (mask_bool.astype(np.uint8) * 255)
    return Image.fromarray(m, mode="L")

class PrefPaintRunner:
    def __init__(self, model_id_or_path: str = "kd5678/prefpaint-v1.0", device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.pipe = AutoPipelineForInpainting.from_pretrained(model_id_or_path).to(device)

    @torch.inference_mode()
    def inpaint_frame(
        self,
        image_rgb_u8: np.ndarray,         # [H,W,3], uint8 RGB
        mask_bool: np.ndarray,            # [H,W], True=foreground to remove
        prompt: str,
        eta: float = 0.7,
        num_inference_steps: int = 25,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        image = _to_pil_rgb(image_rgb_u8)
        mask  = _to_pil_mask(mask_bool)
        out = self.pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            eta=eta,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        return np.array(out, dtype=np.uint8)  # RGB uint8

    @torch.inference_mode()
    def inpaint_sequence(
        self,
        frames_rgb_u8: np.ndarray,        # [T,H,W,3], uint8
        masks_bool: np.ndarray,           # [T,H,W], True=foreground to remove
        prompt: str,
        eta: float = 0.7,
        num_inference_steps: int = 25,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        progress_fn=None,
    ) -> np.ndarray:
        assert frames_rgb_u8.shape[0] == masks_bool.shape[0]
        T = frames_rgb_u8.shape[0]
        out = []
        gen = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        for t in range(T):
            image = _to_pil_rgb(frames_rgb_u8[t])
            mask  = _to_pil_mask(masks_bool[t])
            res = self.pipe(
                prompt=prompt,
                image=image,
                mask_image=mask,
                eta=eta,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=gen,
            ).images[0]
            out.append(np.array(res, dtype=np.uint8))
            if progress_fn:
                try: progress_fn(t, T)
                except Exception: pass
        return np.stack(out, axis=0)  # [T,H,W,3]

    @torch.inference_mode()
    def background_from_reference(
        self,
        ref_frame_rgb_u8: np.ndarray,     # [H,W,3]
        ref_mask_bool: np.ndarray,        # [H,W]
        prompt: str,
        eta: float = 0.7,
        num_inference_steps: int = 25,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Use inpainting on ONE reference frame to synthesize a clean background.
        Return the resulting image; you can then composite it under every frame.
        """
        bg = self.inpaint_frame(
            ref_frame_rgb_u8, ref_mask_bool, prompt,
            eta=eta, num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, seed=seed,
        )
        return bg  # [H,W,3] uint8
