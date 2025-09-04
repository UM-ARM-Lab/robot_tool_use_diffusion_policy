"""
Post-processes raw ROS bag data for Victor robot into a structured dataset for Diffusion Policy training.

This script converts raw ROS bag data (stored as .zarr files) into a processed dataset by:
1. Synchronizing sensor topics at a fixed frequency, OR aligned to vision if vision is enabled
2. (Optional) Interpolating missing values to exact timestamps
3. Extracting robot poses, joint states, gripper status, and wrench data from MotionStatus
4. (Optional) Including vision data from Zivid cameras

Inputs:
- Raw dataset directory containing episode folders with .zarr files and optional image data
- Configuration parameters (robot side, vision enabled, interpolation settings)

Outputs:
- Processed dataset saved as .zarr and .h5 files containing:
  - Synchronized robot observations (joint positions, gripper states, poses)
  - Robot actions (joint commands, gripper commands)
  - Optional image data and wrench force windows
  - Episode metadata and timestamps
"""


from dataclasses import dataclass, fields
from typing import Optional, Union
import os

import numpy as np
from diffusion_policy.victor_data.ros_utils import *

@dataclass
class BagProcessorConfig:
    data_in_dir: str = "<PATH_TO_DATA_IN_DIR>"
    data_out_dir: str = "~/datasets/robotool_processed"
    ln_out_dir: str = "data/victor"

    ds_label: str = "<DATASET_LABEL>"
    processed_path: str = "<PROCESSED_PATH>"

    side: str = "right"
    use_vision: bool = True
    use_wrench: bool = True
    use_aux: bool = False
    use_interpolation: bool = False

    trajectory_filter_regex: Optional[str] = None

    # Chunking
    chunking_start_label: str = "<start>"
    chunking_end_label: str = "<end>"

    no_img_hz: int = 10
    train_split: float = 0.85
    wrench_hist_window: int = 30
    pc_sample_size: int = 4096
    pc_sample_box: Union[np.ndarray, str] = np.array([(-0.4, -0.03), (-0.3, 0.05), (0.6, 1.1)])

    hardcode_zivid_calib_matrix: np.ndarray = np.array([
        [-0.45513538,  0.64372754, -0.6151964,   1.2741948],
        [ 0.86081827,  0.4947717,  -0.11913376,  0.07540844],
        [ 0.22769208, -0.5837943,  -0.77932054,  1.8988546],
        [ 0.,          0.,          0.,          1.        ]
    ])
    recalibrate_pc: bool = False
    recalibrate_matrix = np.array([
        [ 9.99162439e-01,  7.65615027e-04, -4.09125319e-02, -1.61451845e-03],
        [-6.81485391e-04,  9.99997625e-01,  2.07023617e-03, -1.84323175e-03],
        [ 4.09140197e-02, -2.04062093e-03,  9.99160587e-01,  1.97333308e-04],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    zivid_calib_from_tf: bool = False

    def update(self, config: dict):
        """Update dataclass fields from a config dictionary (e.g., YAML)."""
        valid_keys = {f.name for f in fields(self)}

        for key, value in config.items():
            if key not in valid_keys:
                continue
            # Expand environment variables for string values
            if isinstance(value, str):
                value = os.path.expandvars(value)
            # special handling for pc_sample_box
            if key == "pc_sample_box" and isinstance(value, dict):
                # Expect dict with keys 'x', 'y', 'z'
                value = [list(value["x"]), list(value["y"]), list(value["z"])]
                value = np.array(value)
            setattr(self, key, value)

        # Expand environment variables for processed_path calculation
        self.data_out_dir = os.path.expandvars(self.data_out_dir)
        self.data_in_dir = os.path.expandvars(self.data_in_dir)
        
        # Update processed_path
        self.processed_path = os.path.join(self.data_out_dir, self.ds_label)

        return self