import numpy as np
import zarr
import os

from PIL import Image

def read_episode(zarr_path, export_dir=None, export_fmt="jpg"):
    """
    Returns:
      frames: np.ndarray [T,H,W,3] uint8
      meta:   dict with H, W, attrs (zarr root attrs)
    If export_dir is provided, also writes frames to disk.
    """
    root = zarr.open(zarr_path, mode="r")

    # Adjust the key here if your layout differs
    frames = np.asarray(root["data"]["image"])  # [T,H,W,3], uint8

    # Basic sanity / normalization
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected [T,H,W,3], got {frames.shape}")
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)

    meta = {
        "H": frames.shape[1],
        "W": frames.shape[2],
        "attrs": dict(root.attrs),
    }

    # Optional export (replicates your current behavior)
    if export_dir is not None:
        os.makedirs(export_dir, exist_ok=True)
        for i, img in enumerate(frames):
            Image.fromarray(img).save(os.path.join(export_dir, f"{i:05d}.{export_fmt}"))

    return frames, meta

def write_episode(zarr_path, frames, meta):
    root = zarr.open(zarr_path, mode="w")
    g = root.create_group("data")
    g.create_dataset("image", data=frames, chunks=(1,)+frames.shape[1:], dtype=frames.dtype)
    root.attrs.update(meta.get("attrs", {}))
