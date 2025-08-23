import os
from dataclasses import dataclass
from typing import Tuple, Optional, List

import cv2
import numpy as np
from PIL import Image

# ---------- Metadata to reverse/inspect resizing ----------
@dataclass
class LetterboxMeta:
    target_size: Tuple[int, int]  # (H_t, W_t)
    scales: List[Tuple[float, float]]  # per-frame (sy, sx)
    offsets: List[Tuple[int, int]]     # per-frame (y_off, x_off)
    orig_shapes: List[Tuple[int, int]] # per-frame (H, W)

# ---------- Core letterbox helpers ----------
def _letterbox_rgb(img: np.ndarray, target: int) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """Letterbox a single RGB image to (target x target) with black padding."""
    assert img.ndim == 3 and img.shape[2] == 3
    h, w = img.shape[:2]
    scale = min(target / w, target / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target, target, 3), dtype=np.uint8)

    x_off = (target - new_w) // 2
    y_off = (target - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas, (new_h / h, new_w / w), (y_off, x_off)

def _letterbox_mask(mask: np.ndarray, target: int, invert: bool = False) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """Letterbox a single mask to (target x target). Keeps mask binary."""
    assert mask.ndim == 2, "mask must be HxW (grayscale/boolean)"
    if mask.dtype == bool:
        m = mask.astype(np.uint8) * 255
    else:
        m = mask

    if invert:
        m = cv2.bitwise_not(m)

    h, w = m.shape[:2]
    scale = min(target / w, target / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # default background (outside) = 0 for masks; if you prefer 255, change below
    canvas = np.zeros((target, target), dtype=np.uint8)

    x_off = (target - new_w) // 2
    y_off = (target - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    # return binary bool
    return (canvas > 127).astype(np.bool_), (new_h / h, new_w / w), (y_off, x_off)

# ---------- Batch APIs for the pipeline ----------
def resize_batch_rgb(frames: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> Tuple[np.ndarray, LetterboxMeta]:
    """
    frames: [T,H,W,3] uint8 RGB
    returns: frames_resized [T,tH,tW,3], meta
    """
    assert frames.ndim == 4 and frames.shape[-1] == 3, f"Expected [T,H,W,3], got {frames.shape}"
    tH, tW = target_size
    assert tH == tW, "This letterbox function expects square targets"

    out, scales, offsets, shapes = [], [], [], []
    for f in frames:
        lb, (sy, sx), (yo, xo) = _letterbox_rgb(f, tH)
        out.append(lb)
        scales.append((sy, sx))
        offsets.append((yo, xo))
        shapes.append(f.shape[:2])
    return np.stack(out, axis=0), LetterboxMeta(target_size=(tH, tW), scales=scales, offsets=offsets, orig_shapes=shapes)

def resize_batch_masks(masks: np.ndarray, target_size: Tuple[int, int] = (512, 512), invert: bool = False) -> Tuple[np.ndarray, LetterboxMeta]:
    """
    masks: [T,H,W] (bool or uint8)
    returns: masks_resized [T,tH,tW] (bool), meta
    """
    assert masks.ndim == 3, f"Expected [T,H,W], got {masks.shape}"
    tH, tW = target_size
    assert tH == tW, "This letterbox function expects square targets"

    out, scales, offsets, shapes = [], [], [], []
    for m in masks:
        lb, (sy, sx), (yo, xo) = _letterbox_mask(m, tH, invert=invert)
        out.append(lb)
        scales.append((sy, sx))
        offsets.append((yo, xo))
        shapes.append(m.shape[:2])
    return np.stack(out, axis=0), LetterboxMeta(target_size=(tH, tW), scales=scales, offsets=offsets, orig_shapes=shapes)

# ---------- Simple square resize back to original HxW (no letterbox undo; uses standard resize) ----------
def resize_batch_to(frames: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    """Plain resize (no padding), for upscaling back to (H,W)."""
    Ht, Wt = size_hw
    return np.stack([cv2.resize(f, (Wt, Ht), interpolation=cv2.INTER_AREA) for f in frames], axis=0)

# ---------- Optional: folder→folder utilities (mirror your scripts) ----------
def resize_folder_rgb(input_folder: str, output_folder: str, target_size: int = 512) -> None:
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            p = os.path.join(input_folder, filename)
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            out, _, _ = _letterbox_rgb(img, target_size)
            Image.fromarray(out).save(os.path.join(output_folder, filename))

def resize_folder_masks(input_folder: str, output_folder: str, target_size: int = 512, invert: bool = False, bg_value: int = 0) -> None:
    """
    bg_value: background pad value for saved image (0 or 255). Only affects saved file; internal mask stays binary.
    """
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            p = os.path.join(input_folder, filename)
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            lb, _, _ = _letterbox_mask(m, target_size, invert=invert)
            # convert back to uint8 for saving (255 = foreground)
            to_save = (lb.astype(np.uint8) * 255)
            if bg_value == 255:  # make outside white if you prefer
                to_save[~lb] = 255
            cv2.imwrite(os.path.join(output_folder, filename), to_save)

def invert_masks_in_folder(folder_path: str) -> None:
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            p = os.path.join(folder_path, filename)
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to load {filename}")
                continue
            cv2.imwrite(p, cv2.bitwise_not(img))
