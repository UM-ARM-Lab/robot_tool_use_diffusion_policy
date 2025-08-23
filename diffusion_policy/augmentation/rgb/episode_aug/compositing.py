import numpy as np

def composite_with_clean_bg(frames, masks, clean_bg):
    """
    Overlay RGB frames onto a clean background using masks.
    
    Args:
        frames: [T,H,W,3] uint8, RGB images
        masks:  [T,H,W] bool or uint8, True/255 = foreground
        clean_bg: [H,W,3] uint8, RGB background (same for all frames)
    
    Returns:
        composited: [T,H,W,3] uint8
    """
    T, H, W, _ = frames.shape
    if masks.dtype != bool:
        masks = masks > 127
    bg = np.broadcast_to(clean_bg[None, ...], (T, H, W, 3))
    masks3 = masks[..., None]  # [T,H,W,1]
    result = np.where(masks3, frames, bg)
    return result.astype(np.uint8)