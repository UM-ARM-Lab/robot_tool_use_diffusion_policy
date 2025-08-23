#!/bin/bash
# configure which files to process
DATA_IN_DIR="$HOME/datasets/robotool"
DATA_OUT_DIR="$HOME/datasets/robotool_proc"
LN_DIR="./data/victor"

PROCESSED_PATH="$DATA_OUT_DIR/ds_pro_08_09_full102_no_interp.zarr.zip"
POST_PROCESSED_PATH="$DATA_OUT_DIR/victor_data_08_09_full102_no_interp.zarr.zip"

# Processing configuration
SIDE="right"
USE_VISION="true"
USE_WRENCH="true"
SPLIT="false"
USE_AUX="true"
USE_INTERPOLATION="false"
# TODO add a selector for end effector stuff

mamba activate robodiff_vic

python3 diffusion_policy/victor_data/extract_bag_data.py -d $DATA_IN_DIR -s $SPLIT
case $? in
  1) printf "ERROR! %s failed to be parsed! (exit code 1)\n" "$BAG_PATH"
     exit 1;;
  0) printf "SUCCESS! %s parsed into %s (exit code 0)\n" "$BAG_PATH" "$RAW_PATH";;
esac

python3 diffusion_policy/victor_data/postprocess_bag_data.py -p $PROCESSED_PATH -s $SIDE -d $DATA_IN_DIR -v $USE_VISION -a $USE_AUX -i $USE_INTERPOLATION
case $? in
  1) printf "ERROR! %s failed to be processed! (exit code 1)\n" "$DATA_IN_DIR"
     exit 1;;
  0) printf "SUCCESS! %s processed into %s (exit code 0)\n" "$DATA_IN_DIR" "$PROCESSED_PATH";;
esac

python3 diffusion_policy/victor_data/postprocess_zarr.py -p $POST_PROCESSED_PATH -d $PROCESSED_PATH -a $USE_AUX
case $? in
  1) printf "ERROR! %s failed to be processed! (exit code 1)\n" "$PROCESSED_PATH"
     exit 1;;
  0) printf "SUCCESS! %s processed into %s (exit code 0)\n" "$PROCESSED_PATH" "$POST_PROCESSED_PATH";;
esac