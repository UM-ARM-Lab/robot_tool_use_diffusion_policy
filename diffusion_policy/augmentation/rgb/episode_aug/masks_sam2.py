# episode_aug/masks_sam2.py
import os, json, tempfile
import numpy as np
from PIL import Image
import torch
import sys
from IPython import embed
# Add the sam2 directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'sam2'))
from sam2.build_sam import build_sam2_video_predictor
 

def _pick_device():
    if torch.cuda.is_available():
        d = torch.device("cuda")
        # match your notebook defaults
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        return d
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS support note: outputs may differ vs CUDA
        return torch.device("mps")
    return torch.device("cpu")

def sam2_masks_from_clicks(
    frames,                       # np.uint8 [T,H,W,3]
    clicks_json_path,             # str, schema below
    model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
    sam2_checkpoint="checkpoints/sam2.1_hiera_large.pt",
):
    """
    Returns:
        masks: bool np.ndarray of shape [T, H, W]
    JSON schema (clicks_json_path):
    {
      "ann_frame_idx": 0,            # frame index where you clicked
      "ann_obj_id": 1,               # arbitrary integer ID
      "points": [[x,y], [x,y], ...], # pixel coords (float or int)
      "labels": [1,1,1,...]          # 1 = positive, 0 = negative
    }
    """
    
    assert frames.ndim == 4 and frames.shape[-1] == 3 and frames.dtype == np.uint8, \
        f"frames must be [T,H,W,3] uint8, got {frames.shape} {frames.dtype}"

    with open(clicks_json_path, "r") as f:
        ann = json.load(f)
     
    ann = ann['objects'][0]
    ann_frame_idx = int(ann["ann_frame_idx"])
    ann_obj_id    = int(ann.get("ann_obj_id", 1))
    points        = np.array(ann["points"], dtype=np.float32)
    labels        = np.array(ann["labels"], dtype=np.int32)

    device = _pick_device()

    # Change to sam2 directory for config loading
    original_cwd = os.getcwd()
    sam2_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'sam2')
    os.chdir(sam2_dir)
    # embed()
    try:
        # Build predictor
        predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

    # SAM2 expects a directory of sequential JPGs. Create a temp dir and dump frames.
    T, H, W, _ = frames.shape
    with tempfile.TemporaryDirectory() as td:
        # write frames as 00000.jpg, 00001.jpg, ...
        for i in range(T):
            Image.fromarray(frames[i]).save(os.path.join(td, f"{i:05d}.jpg"))

        # init & reset state
        inference_state = predictor.init_state(video_path=td)
        predictor.reset_state(inference_state)

        # seed clicks
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

        # optional: you can visualize or verify here if needed

        # propagate over the whole sequence
        masks = np.zeros((T, H, W), dtype=bool)
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            # assume single object with id = ann_obj_id; if multiple objs, you can OR them
            if len(out_obj_ids) == 0:
                continue
            # find the index of our ann_obj_id (fallback to the first if not present)
            if ann_obj_id in out_obj_ids.tolist():
                i = out_obj_ids.tolist().index(ann_obj_id)
            else:
                i = 0
            m = (out_mask_logits[i] > 0.0).detach().cpu().numpy()  # [H,W] bool
            masks[out_frame_idx] = m

    return masks  # [T,H,W] bool
