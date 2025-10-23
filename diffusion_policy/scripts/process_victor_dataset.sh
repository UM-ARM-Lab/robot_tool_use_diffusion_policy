#!/bin/bash
# configure which files to process
config_file=${HOME}/datasets/robotool_configs/09_08_cap_121_138_screw_off.yaml

python3 diffusion_policy/victor_data/extract_bag_data.py -c $config_file --version_only

# python3 diffusion_policy/victor_data/postprocess_bag_data.py -p $PROCESSED_PATH -s $SIDE -d $DATA_IN_DIR -v $USE_VISION -a $USE_AUX -i $USE_INTERPOLATION
python3 diffusion_policy/victor_data/postprocess_bag_data.py --config $config_file