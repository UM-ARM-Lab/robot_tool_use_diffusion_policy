# episode_aug/pipeline.py
from .io_zarr import read_episode, write_episode
# Pick ONE of these based on your JSON format:
from .masks_sam2 import sam2_masks_from_clicks as sam2_masks
# from .masks_sam2 import sam2_masks_from_json as sam2_masks

from .resize import resize_batch_rgb, resize_batch_masks, resize_batch_to
from .compositing import composite_with_clean_bg
# from .prefpaint_bg import PrefPaintRunner  # using the diffusers-based adapter
from IPython import embed


def run_pipeline(cfg):
    # 1) Read episode
    frames, meta = read_episode(cfg["input_zarr"])  # [T,H,W,3] uint8
    
    # 2) SAM2 masks (bool [T,H,W])
    # If using clicks JSON, ensure you call sam2_masks_from_clicks and pass cfg paths.
    masks = sam2_masks(frames, cfg["sam2_json"])
    embed()
    # 3) Resize to working size
    wh = tuple(cfg.get("work_size", [512, 512]))
    frames_512, _ = resize_batch_rgb(frames, target_size=wh)
    masks_512, _  = resize_batch_masks(masks, target_size=wh, invert=False)  # True=invert if needed

    # 4/5/6) PrefPaint: either full sequence or single background + composite
    pp_cfg = cfg.get("prefpaint", {})
    mode = pp_cfg.get("mode", "background")  # "background" | "sequence"
    runner = PrefPaintRunner(
        model_id_or_path=pp_cfg.get("model", "kd5678/prefpaint-v1.0"),
        device=pp_cfg.get("device", "cuda")
    )

    if mode == "sequence":
        # Inpaint every frame (slow, highest quality)
        comp_512 = runner.inpaint_sequence(
            frames_512, masks_512,
            prompt=pp_cfg["prompt"],
            eta=pp_cfg.get("eta", 0.7),
            num_inference_steps=pp_cfg.get("steps", 25),
            guidance_scale=pp_cfg.get("guidance_scale", None),
            seed=pp_cfg.get("seed", None),
        )
    else:
        # "background" (fast): make one clean BG then composite under all frames
        idx = pp_cfg.get("ref_idx", 0)
        bg_512 = runner.background_from_reference(
            frames_512[idx], masks_512[idx],
            prompt=pp_cfg["prompt"],
            eta=pp_cfg.get("eta", 0.7),
            num_inference_steps=pp_cfg.get("steps", 25),
            guidance_scale=pp_cfg.get("guidance_scale", None),
            seed=pp_cfg.get("seed", None),
        )

        ## pause and save the bg image
        ## then go to io-paint lama and remove the foreground
        ## save the cleaned bg image
        ## then resume here and composite
        clean_bg = cv2.imread("clean_bg.png")
        comp = composite_with_clean_bg(frames, masks, clean_bg)

        # Optional post inpaint of seams (cheap, classical CV)
        if cfg.get("inpaint", {}).get("enabled", True):
            comp_512 = inpaint_sequence(
                comp_512, masks_512,
                method=cfg["inpaint"].get("method", "telea")
            )

    # 7) Resize back to original resolution
    comp_full = resize_batch_to(comp_512, (frames.shape[1], frames.shape[2]))

    # 8) Save
    write_episode(cfg["output_zarr"], comp_full, meta)
