#!/bin/bash
# configure which files to process
DATA_IN_DIR="$HOME/datasets/robotool_hdd"
DATA_OUT_DIR="$HOME/datasets/robotool_processed"
LN_OUT_DIR="data/victor"

DS_LABEL="08_27_cap_106_full"
PROCESSED_PATH="$DATA_OUT_DIR/victor_${DS_LABEL}"

# Processing configuration
SIDE="right"
USE_VISION="true"
USE_WRENCH="true"
SPLIT="false"
USE_AUX="true"
USE_INTERPOLATION="false"
# TODO add a selector for end effector stuff

# Check if LN_OUT_DIR variable exists and path exists, then create symlink if not present
if [[ -n "$LN_OUT_DIR" && -d "$DATA_OUT_DIR" ]]; then
    if [[ ! -L "$LN_OUT_DIR" ]]; then
        ln -s "$DATA_OUT_DIR" "$LN_OUT_DIR"
        echo "Symlink created: $LN_OUT_DIR -> $DATA_OUT_DIR"
    else
        echo "Symlink already exists: $LN_OUT_DIR"
    fi
fi

# python3 diffusion_policy/victor_data/extract_bag_data.py -d $DATA_IN_DIR -s $SPLIT
# case $? in
#   1) printf "ERROR! %s failed to be parsed! (exit code 1)\n" "$BAG_PATH"
#      exit 1;;
#   0) printf "SUCCESS! %s parsed into %s (exit code 0)\n" "$BAG_PATH" "$RAW_PATH";;
# esac

python3 diffusion_policy/victor_data/postprocess_bag_data.py -p $PROCESSED_PATH -s $SIDE -d $DATA_IN_DIR -v $USE_VISION -a $USE_AUX -i $USE_INTERPOLATION
case $? in
  1) printf "ERROR! %s failed to be processed! (exit code 1)\n" "$DATA_IN_DIR"
     exit 1;;
  0) printf "SUCCESS! %s processed into %s (exit code 0)\n" "$DATA_IN_DIR" "$PROCESSED_PATH";;
esac
