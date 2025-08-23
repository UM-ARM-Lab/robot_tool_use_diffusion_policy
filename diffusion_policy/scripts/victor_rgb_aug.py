import argparse, yaml
from utils.augmentation.rgb.pipeline import run_pipeline

# TODO: 
# 1. Read an episode zarr file
# 2. Get masks using SAM2 with json file
# 3. Resize masks to 512 x 512 and rgb images to 512 x 512 and save 
# 4. Use PrefPaint with just 1 iteration to get a background
# 5. Remove the foreground 
# 6. Inpaint the entire trajectory with the new background
# 7. Resize back to original size 
# 8. Save the new augmented episode zarr file with the augmented images 



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="config/config.yaml")
    args = p.parse_args()
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    run_pipeline(cfg)

if __name__ == "__main__":
    main()
