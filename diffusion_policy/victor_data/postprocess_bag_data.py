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
import argparse
import re
import os

import zarr
import h5py

from rclpy.time import Time
from tf2_ros.buffer import Buffer
import tf2_py as tf2

import numpy as np
import tqdm
import json
import yaml

# TODO NOTE: every time you want to run this script, make sure to 
#               source ros/install/setup.bash
#            (after building victor_hardware_interfaces of course)
# from victor_hardware_interfaces.msg import MotionStatus, Robotiq3FingerStatus

from diffusion_policy.victor_data.ros_utils import *
from diffusion_policy.victor_data.data_utils import (
    read_zarr_dict_recursive,
    store_h5_dict,
    store_h5_dict_a,
    SmartDict,
    store_zarr_dict_diff_data,
    store_zarr_dict_diff_data_a
)

from diffusion_policy.victor_data.bag_process_config import BagProcessorConfig

class BagPostProcessor():
    def __init__(self,
        config: BagProcessorConfig
    ):
        self.config = config
        print("Initializing BagPostProcessor with config:", config)
        self.wrench_hist_window = int(config.wrench_hist_window)    # number of wrench messages per window
        self.freq = 1/config.no_img_hz # NOTE: timestamps are in ns   # TODO no longer needed due to the zivid times

        self.use_vision = config.use_vision # should the processed ds include images or not
        self.use_aux_learn = config.use_aux
        self.use_interpolation = config.use_interpolation
        self.tracked_sides = [config.side]

        # dataset storage
        self.data_dir = config.data_in_dir
        self.proc_dataset = SmartDict(backend="numpy")  # processed dataset
        self.train_split = config.train_split
        self.recalibrate_pc = config.recalibrate_pc
        self.last_train_ep_end = 0
        self.last_val_ep_end = 0

        # Filter trajectories
        self.ep_dirs = self._filter_trajectory_by_name()
        self.num_traj = len(self.ep_dirs)
        self._train_test_split()

        # Setup pbar
        self.pbar = tqdm.tqdm(self.ep_dirs, desc="Processing episodes")
        
        # Setup files
        processed_path = config.processed_path
        self.train_zarr_fn = processed_path + "_train.zarr.zip"
        self.train_h5_fn = processed_path + "/training/ds.h5"
        self.val_zarr_fn = processed_path + "_val.zarr.zip"
        self.val_h5_fn = processed_path + "/validation/ds.h5"

        os.makedirs(processed_path + "/training", exist_ok=True)
        os.makedirs(processed_path + "/validation", exist_ok=True)

        self._setup_pc_processing()

    def _filter_trajectory_by_name(self):
        # Filter out trajectories
        # Check for empty source
        all_ep_dirs = sorted(os.listdir(self.data_dir))  # Sort for consistent display
        num_raw_epi = len(all_ep_dirs)
        if num_raw_epi < 2:
            raise ValueError("Error: Not enough raw episodes found in data_in_dir, found:", num_raw_epi)
        
        # Display episode filtering in compact block format
        symbols_per_line = 20  # Number of symbols per line
        
        if self.config.trajectory_filter_regex is None:
            return all_ep_dirs

        ep_dirs = []
        matched_symbols = []
        print(f"Episode filtering with regex: {self.config.trajectory_filter_regex}")
        
        for ep_dir in all_ep_dirs:
            if not re.match(self.config.trajectory_filter_regex, ep_dir):
                matched_symbols.append("✗")
            else:
                matched_symbols.append("✓")
                ep_dirs.append(ep_dir)
        
        # Display symbols in blocks
        for i in range(0, len(matched_symbols), symbols_per_line):
            chunk_symbols = matched_symbols[i:i + symbols_per_line]
            symbols_line = f"{i} "
            symbols_line += "".join(chunk_symbols)
            print(f"  {symbols_line}")

        matched_count = len(ep_dirs)
        filtered_count = num_raw_epi - matched_count
        print(f"  {matched_count} matched (✓), {filtered_count} filtered out (✗)")
        return ep_dirs
    
    def _train_test_split(self):
        # Sample train/val trajectories
        self.train_epi = np.random.choice(
            self.num_traj, size=int(self.num_traj*self.train_split), replace=False
        )
        print(f"Selected {len(self.train_epi)} training episodes out of {self.num_traj}")
        self.val_epi = np.setdiff1d(np.arange(self.num_traj), self.train_epi)
        if self.train_epi.shape[0] < 1:
            self.train_epi = np.array([0])
        if self.val_epi.shape[0] < 1:
            self.val_epi = np.array([0])
        # Cache the minimum blocks
        self.train_epi0 = self.train_epi.min()
        self.val_epi0 = self.val_epi.min()
        print(f"Train with {len(self.train_epi)} and test with {len(self.val_epi)} episodes")

    def _setup_pc_processing(self):
        # Setup point cloud processing
        self.pc_sample_size = self.config.pc_sample_size
        assert len(self.config.pc_sample_box) == 3
        for dim in self.config.pc_sample_box:
            assert len(dim) == 2
        self.pc_sample_box = self.config.pc_sample_box

    def post_process(self):
        """Main post process function"""
        for ep_i, ep_dir in enumerate(self.pbar):
            print(f"Processing {ep_i+1}/{self.num_traj}: {os.path.basename(ep_dir)}")
            self.proc_dataset = SmartDict(backend="numpy")
            try:
                self.post_process_ep(self.data_dir, ep_dir, ep_i)
            except Exception as e:
                # import traceback
                print(f"Error processing {ep_dir}: {e}")
                # print("Full traceback:")
                # traceback.print_exc()
                continue

        # Finally, copy the validation set as the debug set for h5
        debug_h5_fn = self.config.processed_path + "/debug/ds.h5"
        os.makedirs(self.config.processed_path + "/debug", exist_ok=True)
        os.system(f"cp {self.val_h5_fn} {debug_h5_fn}")

        print(f"saving processed trajectories to", self.train_zarr_fn)

    def post_process_ep(self, data_dir, ep_name, ep_i):
        # Setup
        self.zarr_dir = os.path.join(data_dir, ep_name, "raw", ep_name + ".zarr.zip")
        self.img_dir = os.path.join(data_dir, ep_name, "zivid")

        # forcing the buffer to keep all the updates (up to an hr)
        self.buffer = Buffer(cache_time=Duration(seconds=3600))

        # Read data
        zf = zarr.open(self.zarr_dir, mode='r')
        self.raw_dataset = SmartDict(backend="numpy")
        read_zarr_dict_recursive(zf, self.raw_dataset)

        # Setup annotation file
        annot_file = os.path.join(data_dir, ep_name, "annotation.json")
        try:
            with open(annot_file, "r") as f:
                self.annotation = json.load(f)
        except Exception as e:
            # print(f"Error loading annotation file {annot_file}: {e}")
            self.annotation = None

        # Prepare for processing
        self._setup_topics(zf)
        self._setup_timestamps()
        # Setup display
        new_start = self.timestamps[0] // 1e9
        new_end = self.timestamps[-1] // 1e9
        self.pbar.set_description(f"Epi {ep_i+1} | {new_start}-{new_end}s")

        # Actual processing
        self._register_transform_buffer()
        self._interpolate_tf()
        self._add_other_values()

        self._add_vision()
        self._add_finger_specifics()
        self._add_wrench()
        self._add_progress()
        self._add_extra_fields()
        self._add_episode_ends(ep_i)
        
        # Then save
        zarr_fname = self.train_zarr_fn if ep_i in self.train_epi else self.val_zarr_fn
        h5_fname = self.train_h5_fn if ep_i in self.train_epi else self.val_h5_fn

        # print(self.train_epi0, ep_i)
        if ep_i == self.train_epi0 or ep_i == self.val_epi0:   # TODO ugly :(
            store_zarr_dict_diff_data(zarr_fname, self.proc_dataset)
            store_h5_dict(h5_fname, self.proc_dataset)
        else:
            store_zarr_dict_diff_data_a(zarr_fname, self.proc_dataset)
            store_h5_dict_a(h5_fname, self.proc_dataset)

    def _setup_topics(self, zf):
        """
        Setup which topics to track
        """
        self.parent_topics = [  # mostly used for finding the corresponding timestamp for these topics
            "gripper_status",
            "motion_status",
            "wrench",
            "joint_angles",
        ]
        
        # must be included in both b/c other code depends on it
        manual_parent_topics = ["gripper_status", "motion_status"]

        # dict of all topics that we are tracking and their value topics (like data or orientation or anything else)
        self.tracked_topics = {}
        for side in self.tracked_sides:
            # automatically process parent topics's subtopics
            for pt in self.parent_topics:
                if pt in manual_parent_topics:
                    continue

                key = "/" + side + "_arm/" + pt
                if key + "/timestamp" not in self.raw_dataset.keys():
                    continue    # no data was recorded for this topic

                self.tracked_topics[key] = [t for t in zf[key].keys() if "timestamp" not in t]
                # TODO removed the per arm tracking
                # self.tracked_topics[pt] = [t for t in get_h5_leaf_topics(hf, key, 0, 1) if "timestamp" not in t]
                # self.tracked_topics.extend()

            self.tracked_topics["/" + side + "_arm/gripper_status"] = ["finger_a_status", "finger_b_status", "finger_c_status", "scissor_status"]
            # measured joint position is used for the obs, while joint_angles are the commands
            self.tracked_topics["/" + side + "_arm/motion_status"] = ["measured_joint_position"] #, "commanded_joint_position"]

            # manually adding topics that do not have a left and a right side
            # print("Tracking", self.tracked_topics)

            # contains all the wrench topics that should be tracked at a higher frequency
            self.wrench_topics = ["/" + side + "_arm/wrench"]   # NOTE ASSUMES THAT THIS TOPIC HAS /data

            # contains all the transform frames in the dataset,
            #  not necessarily all that will be included in the final dataset
            self.tf_buffer_topics = []
            for side in self.tracked_sides:
                self.tf_buffer_topics.extend([f"/{side}_arm/pose/" + k for k in zf[f"/{side}_arm/pose"].keys()])
                self.tf_buffer_topics.extend([f"/{side}_arm/pose_static/" + k for k in zf[f"/{side}_arm/pose_static"].keys()])
            self.tf_buffer_reference_topics = ["/reference_pose/" + k for k in zf["/reference_pose"].keys()]

    # finds the earliest time when at all tracked topics have published at least 1 msg
    def __find_earliest_valid_time(self):
        """
        Find the earliest time at which all relevant topics have published at least one message.
        """
        # Generate all possible keys for parent topics across all tracked sides
        ks = []
        ks.extend(["/" + side + "_arm/" + t + "/timestamp" for side in self.tracked_sides \
                   for t in self.parent_topics])
        ks.extend(["/" + side + "_arm/pose/" + t + "/timestamp" for side in self.tracked_sides \
                   for t in self.__get_tf_topics(side)])
        ks.extend(["/" + side + "_arm/pose_static/" + t + "/timestamp" for side in self.tracked_sides \
                   for t in self.__get_tf_topics(side)])
        ks = [k for k in ks if k in self.raw_dataset.keys()]
        earliest_ts_each = [self.raw_dataset[k][0] for k in ks]
        earliest_ts = max(earliest_ts_each) if earliest_ts_each else 0.0
        return earliest_ts
    
    def _setup_timestamps(self):
        ### SETUP timestamps
        evt = self.__find_earliest_valid_time()
        if self.use_vision:
            evt_zivid = np.searchsorted(self.raw_dataset["/zivid/timestamp"], evt)
            self.timestamps = self.raw_dataset["/zivid/timestamp"][evt_zivid:]
        else:
            self.timestamps = list(range(evt, int(self.raw_dataset["/duration"][0]), int(self.freq * 1e9)))
        
        print(f"Raw timestamps, from {self.timestamps[0]//1e9} to {self.timestamps[-1]//1e9}s, total {len(self.timestamps)}")
        self._filter_timestamps_for_chunk()
        self._filter_timestamps_for_plateau()
        # Add filtered timestamp to dataset
        for ts in self.timestamps:
            self.proc_dataset.add("data/timestamp", ts)

    def _get_annotation_keyframe_time(self, annotation, target_keyframe, type='start') -> int:
        # Check annotation
        if annotation is None:
            raise ValueError("Annotation file is required for chunking but not found.")
        custom_keyframes = annotation["keyframes"]
        assert isinstance(custom_keyframes, dict), "Keyframes must be a dictionary."

        if type == 'start':
            if target_keyframe.startswith('<start>'):
                return self._zivid_start_ts
            if self.config.chunking_start_label not in custom_keyframes.keys():
                raise ValueError("Chunking start label not found in keyframes.")
        elif type == 'end':
            if target_keyframe.startswith('<end>'):
                return self._zivid_end_ts
            if self.config.chunking_end_label not in custom_keyframes.keys():
                raise ValueError("Chunking end label not found in keyframes.")

        target_ts = int(custom_keyframes[target_keyframe]) * 1e9  # seconds
        assert target_ts <= self._zivid_end_ts, "Target timestamp exceeds Zivid end timestamp."
        return int(target_ts)

    def _filter_timestamps_for_chunk(self):
        """
        Filter timestamps for each chunk of data.
        """
        if not self.use_vision:
            print("Chunking only supported with vision data, skipping chunking.")
            return
        
        if len(self.timestamps) <= 1:
            return
        
        # Setup zivid timestamps
        zivid_timestamps = self.raw_dataset["/zivid/timestamp"]     # sec * 1e9 (ns in int)
        self._zivid_start_ts = int(zivid_timestamps[0])
        self._zivid_end_ts = int(zivid_timestamps[-1])
        
        # If full, just return
        if self.config.chunking_end_label == '<end>' \
            and self.config.chunking_start_label == '<start>':
            return

        start_time = self._get_annotation_keyframe_time(
            self.annotation, self.config.chunking_start_label, type='start'
        )
        end_time = self._get_annotation_keyframe_time(
            self.annotation, self.config.chunking_end_label, type='end'
        )
        # Filter timestamps
        self.timestamps = [ts for ts in self.timestamps if start_time <= int(ts) <= end_time]
        # print(f"Chunking timestamps to {len(self.timestamps)} between {start_time//1e9}-{end_time//1e9}s")

    def _filter_timestamps_for_plateau(self):
        """
        Detect plateaus in the robot's action commands over the recorded timestamps.
        A plateau is defined as a sequence of consecutive timestamps where the robot's
        actions remain unchanged. This method updates the internal timestamp list to
        exclude these plateau regions, which are not informative for training.
        """
        if len(self.timestamps) <= 1:
            return
        
        # Get joint angles data using _interpolate_other_values
        joint_angles = None
        gripper_requests = None
        
        ts_all = np.asarray(self.timestamps)
        do_linear = bool(self.use_interpolation)
        
        # Get joint angles data
        for t, v in self.tracked_topics.items():
            if "joint_angles" in t:
                topic, value_topics = t, v
                break
        ts_arr = np.asarray(self.raw_dataset[topic + "/timestamp"])
        rtn_dict = self._interpolate_other_values(topic, ts_arr, ts_all, do_linear, value_topics)
        if "data/joint_angles_data" in rtn_dict:
            joint_angles = np.array(rtn_dict["data/joint_angles_data"])

        # Create a temporary processed dataset to get gripper data
        temp_proc_dataset = {}
        
        # Get gripper status data
        for t, v in self.tracked_topics.items():
            if "gripper_status" in t:
                topic, value_topics = t, v
                break
        ts_arr = np.asarray(self.raw_dataset[topic + "/timestamp"])
        rtn_dict = self._interpolate_other_values(topic, ts_arr, ts_all, do_linear, value_topics)
        temp_proc_dataset = rtn_dict
        
        # Extract gripper request data from the temporary dataset
        if temp_proc_dataset:
            diffs = ["finger_a", "finger_b", "finger_c", "scissor"]
            finger_topics = [f"data/gripper_status_{d}_status" for d in diffs]
            
            # Check if all finger topics are present
            if all(topic in temp_proc_dataset for topic in finger_topics):
                series = [np.asarray(temp_proc_dataset[topic]) for topic in finger_topics]
                req_arrays = [s[:, 0] for s in series]  # Extract request values (column 0)
                gripper_requests = np.column_stack(req_arrays)  # (T, 4)
        
        if joint_angles is None or gripper_requests is None:
            print("Warning: Could not compute joint angles or gripper data for plateau filtering")
            return
        
        # Create timestamp mask array (1 = keep, 0 = remove)
        mask = np.ones(len(self.timestamps), dtype=bool)
        tolerance = 1e-5
        
        # Compare each timestamp with the previous one
        for i in range(1, len(self.timestamps)):
            # Check if joint angles haven't changed
            joint_unchanged = np.allclose(joint_angles[i], joint_angles[i-1], atol=tolerance)
            gripper_unchanged = np.allclose(gripper_requests[i], gripper_requests[i-1], atol=tolerance)
            
            # If both haven't changed, mark for removal
            if joint_unchanged and gripper_unchanged:
                mask[i] = False
        
        # Update timestamps to keep only non-plateau points
        # original_count = len(self.timestamps)
        self.timestamps = [self.timestamps[i] for i in range(len(self.timestamps)) if mask[i]]
        # filtered_count = len(self.timestamps)
        # print(f"Filtered {original_count - filtered_count} plateau timestamps, kept {filtered_count}")               

    # returns the topics we want to track in the final dataset
    def __get_tf_topics(self, side: str):
        return [
            "target_victor_" + side + "_tool0",
            "victor_" + side + "_tool0",
        ]

    # add a tf topic to the tf buffer
    def __process_tf_topic(self, key):
        """
        Processes a specific TF (transform) topic from the raw dataset and updates the transform buffer.
        This method iterates through the timestamps of a given TF topic, retrieves the associated
        frame IDs, translation, and rotation data, and constructs a transform object. The transform
        is then added to the buffer as either a static or dynamic transform based on the topic name.
        
        Args:
            key (str): The key representing the TF topic in the raw dataset.
        Notes:
            - If the topic contains "static" or "reference" in its name, the transform is treated as static.
            - Assumes that the raw dataset contains the necessary keys for timestamps, frame IDs, 
              translation, and rotation.
        """
        # in case some data was not recorded (like one of the arms hadnt been used)
        # if (key + "/timestamp") not in self.raw_dataset.keys(): return
        
        # key = "/right_arm/pose/" + key
        # print("tkey", key)

        for i in range(len(self.raw_dataset[key + "/timestamp"])):
            frame_id = str(self.raw_dataset[key + "/parent_frame_id"][0])         
            child_frame_id = str(self.raw_dataset[key + "/child_frame_id"][0])
            # print(frame_id, child_frame_id)           
            stamp = self.raw_dataset[key + "/timestamp"][i]
            translation = self.raw_dataset[key + "/translation"][i]
            rotation = self.raw_dataset[key + "/rotation"][i]
            tr = np_arrs_to_tf_transform(
                frame_id,
                child_frame_id,
                stamp,
                translation, 
                rotation
            )
            if "static" in key or "reference" in key:
                self.buffer.set_transform_static(tr, "default_authority")
            else:
                self.buffer.set_transform(tr, "default_authority")

    # fill the buffer with the available transforms (to use for interpolation)
    def _register_transform_buffer(self):
        """
        Registers all truly recorded TF transforms into the transform buffer.
        This includes both dynamic and static transforms from the raw dataset.
        """
        # Combine tf buffer topics and reference topics into a temporary set
        all_tf_topics = set(self.tf_buffer_topics + self.tf_buffer_reference_topics)
        for k in all_tf_topics:
            self.__process_tf_topic(k)

    def _interpolate_tf(self):
        """
        Compute interpolations given recorded TF frames
        """
        # TODO vestigial remnants of when we used to track this
        tf_topics = []
        for side in self.tracked_sides:
            tf_topics.extend(self.__get_tf_topics(side))

        for i, ts in enumerate(self.timestamps):
            #   1. Go through the transformations and use the buffer to look up the state at that time
            for tf_t in tf_topics:
                try:
                    tr = bag_proc.buffer.lookup_transform("victor_root", tf_t, Time(nanoseconds=ts))
                    self.proc_dataset.add("pose/" + tf_t + "_translation", ros_msg_to_arr(tr.transform.translation))
                    self.proc_dataset.add("pose/" + tf_t + "_rotation", ros_msg_to_arr(tr.transform.rotation))
                # except tf2.LookupException: pass # if the topic has no data, ignore (happens when only one of the arms was used)
                except tf2.ExtrapolationException:  # TODO if the topic has run out of the data, fill the rest with the last known value
                    tr = bag_proc.buffer.lookup_transform("victor_root", tf_t, Time()) # get latest message
                    self.proc_dataset.add("pose/" + tf_t + "_translation", ros_msg_to_arr(tr.transform.translation))
                    self.proc_dataset.add("pose/" + tf_t + "_rotation", ros_msg_to_arr(tr.transform.rotation))
                    # ALTERNATIVELY do the latest data from the dict
                    # self.proc_dataset.add("pose/" + topic + "/translation", self.proc_dataset["pose/" + topic  + "/translation"][-1])
                    # self.proc_dataset.add("pose/" + topic + "/rotation", self.proc_dataset["pose/" + topic  + "/rotation"][-1])

        # Deal specifically zivid optical frame

    def _get_static_transform(self, tf_t):
        tr = bag_proc.buffer.lookup_transform("victor_root", tf_t, Time(nanoseconds=self.timestamps[0]))
        translation = ros_msg_to_arr(tr.transform.translation)
        rotation = ros_msg_to_arr(tr.transform.rotation)
        return translation, rotation

    def _compute_bounds(self, ts_arr: np.ndarray, targets: np.ndarray):
        """
        Given a sorted ts_arr and arbitrary targets, return vectorized low/high indices and masks.
        low_i  = index of last ts <= target  (clipped to [0, n-1])
        high_i = index of first ts >= target (clipped to [0, n-1])
        mask_left  : target < ts_arr[0]
        mask_right : target > ts_arr[-1]
        """
        n = len(ts_arr)
        high_raw = np.searchsorted(ts_arr, targets, side="left")
        low_raw = high_raw - 1

        mask_left = (low_raw < 0)
        mask_right = (high_raw >= n)

        low_clip = np.clip(low_raw, 0, n - 1)
        high_clip = np.clip(high_raw, 0, n - 1)
        return low_clip, high_clip, mask_left, mask_right


    def _sample_series(self, 
        ts_arr: np.ndarray,
        y: np.ndarray,
        targets: np.ndarray,
        *,
        mode: str = "linear",
        left_policy: str = "first",   # or "nan"
        right_policy: str = "last"    # or "nan"
    ) -> np.ndarray:
        """
        Vectorized sampling of a time series y(ts_arr) at arbitrary targets.
        - ts_arr: timestamps of ROS data
        - y: values to be sampled
        - targets: target timestamps to get sample from
        Supports:
        - mode="linear" : point-slope interpolation between bounding samples
        - mode="left"   : step function (last-known value on the left)
        Extrapolation:
        - left_policy: 'first' or 'nan'
        - right_policy: 'last' or 'nan'
        Shapes:
        - ts_arr: (N,)
        - y: (N, ...)  (can be multi-dim per timestep)
        - targets: (M,)
        - returns: (M, ...)  matching y’s trailing dims
        """
        low_i, high_i, mask_left, mask_right = self._compute_bounds(ts_arr, targets)
        y_low = y[low_i]
        if mode == "left":
            out = y_low.copy()
        else:
            # linear interpolation
            x1 = ts_arr[low_i]
            x2 = ts_arr[high_i]
            dx = x2 - x1
            same = (dx == 0)

            # prepare broadcasting-friendly slope dtype
            slope_dtype = np.result_type(y_low, np.float64)
            slope = np.zeros_like(y_low, dtype=slope_dtype)
            valid = ~same
            if np.any(valid):
                # Broadcast (y_high - y_low) / dx over trailing dims
                slope[valid] = (y[high_i][valid] - y_low[valid]) / dx[valid, ...]
            out = y_low + slope * (targets - x1)[(...,) + (None,) * (y.ndim - 1)]

            # fall back where x2==x1
            if np.any(same):
                out[same] = y_low[same]

        # Extrapolation policies
        if right_policy == "last":
            out[mask_right] = y[-1]
        elif right_policy == "nan":
            out[mask_right] = np.nan

        if left_policy == "first":
            out[mask_left] = y[0]
        elif left_policy == "nan":
            out[mask_left] = np.nan

        return out

    def _interpolate_other_values(self, topic, ts_arr, ts_all, do_linear, value_topics):
        """
        Compute interpolated or not,
        used by both regular processing AND plateau removal
        """
        rtn = {}
        for value_topic in value_topics:
            key = f"{topic}/{value_topic}"
            s_key = f"data/{topic.split('/')[-1]}_{value_topic}"
            if key not in self.raw_dataset:
                print("ouch", key, "missing")
                continue
            y = np.asarray(self.raw_dataset[key])
            mode = "linear" if do_linear else "left"
            out = self._sample_series(
                ts_arr, y, ts_all,
                mode=mode,
                left_policy="first",
                right_policy="last",
            )
            rtn[s_key] = [row for row in out]
        return rtn

    def _add_other_values(self):
        """
        Vectorized interpolation/left-hold for all tracked topics & values.
        """
        ts_all = np.asarray(self.timestamps)
        do_linear = bool(self.use_interpolation)

        for topic, value_topics in self.tracked_topics.items():
            ts_arr = np.asarray(self.raw_dataset[topic + "/timestamp"])
            rtn_dict = self._interpolate_other_values(topic, ts_arr, ts_all, 
                                                      do_linear, value_topics)
            # Add to dict
            for s_key, out in rtn_dict.items():
                self.proc_dataset[s_key] = out

    def _process_pc(self, pc):
        """
        Filter point cloud to only include points within the specified box.
        pc: (3, N) numpy array
        pc_sample_box: [[x_min, y_min], [x_max, y_max], [z_min, z_max]]
        """

        if self.recalibrate_pc:
            pc[:3] = self.config.recalibrate_matrix[:3,:3] @ pc[:3] + self.config.recalibrate_matrix[:3,3:4]

        # Filter
        x_min, x_max = self.pc_sample_box[0]
        y_min, y_max = self.pc_sample_box[1]
        z_min, z_max = self.pc_sample_box[2]

        mask = (
            (pc[0] >= x_min) & (pc[0] <= x_max) &
            (pc[1] >= y_min) & (pc[1] <= y_max) &
            (pc[2] >= z_min) & (pc[2] <= z_max)
        )
        pc_boxed = pc[:, mask]

        # Then, get transform and transform PC
        if self.config.zivid_calib_from_tf:
            # EITHER, use existing transform in the dataset
            zivid_translation, zivid_rotation = self._get_static_transform("zivid_optical_frame")

            # Build rotation matrix
            from scipy.spatial.transform import Rotation
            rot = Rotation.from_quat(zivid_rotation)
            rot_mat = rot.as_matrix()
            # Build homogeneous matrix
            homo_matrix = np.eye(4)
            homo_matrix[:3, :3] = rot_mat
            homo_matrix[:3, 3] = zivid_translation
        else:
            homo_matrix = self.config.hardcode_zivid_calib_matrix

        # Transform points
        homo_ones = np.ones((1,)+pc_boxed.shape[1:], dtype=pc_boxed.dtype)
        pc_xyz_homo = np.vstack([pc_boxed[:3], homo_ones])      # 4xN
        # print("PC HOMO sample:", pc_xyz_homo[...,:3].T)
        pc_xyz_transformed = homo_matrix @ pc_xyz_homo       # 4xN
        pc_xyz = pc_xyz_transformed[:3]     # 3xN
        pc_boxed[:3] = pc_xyz
        
        sampled_idx = np.random.choice(pc_boxed.shape[1], self.pc_sample_size, replace=True)
        return pc_boxed[:, sampled_idx].T        # (N, 6)

    def _add_vision(self):
        """
        Add Zivid camera data to the processed dataset.
        """
        if not self.use_vision:
            return
        
        img_h5 = None
        for ts in self.timestamps:
            img_i = np.searchsorted(self.raw_dataset["/zivid/timestamp"], ts) 
            fid = self.raw_dataset["/zivid/frame_id"][img_i]
            chunk_id = fid // 50
            offset_id = fid % 50
            if img_h5 is None or offset_id == 0:   # we've finished one chunk -> load next
                img_h5 = h5py.File(os.path.join(self.img_dir, "processed_chunk_" + str(chunk_id) + ".h5"))
            try:
                self.proc_dataset.add("data/image", img_h5["rgb"][offset_id])
                pc = self._process_pc(img_h5["pc"][offset_id])
                if pc.shape[0] <= 0:
                    raise Exception("No points in point cloud after filtering")
                self.proc_dataset.add("data/pc_xyz", pc[:,:3])
                self.proc_dataset.add("data/pc_rgb", pc[:,3:])
            except IndexError:
                print("index", offset_id,"in chunk", chunk_id, "is out of bounds, using last image")
                self.proc_dataset.add("data/image", img_h5["rgb"][-1])
                pc = self._process_pc(img_h5["pc"][-1])
                if pc.shape[0] <= 0:
                    raise Exception("No points in point cloud after filtering")
                self.proc_dataset.add("data/pc_xyz", pc[:,:3])
                self.proc_dataset.add("data/pc_rgb", pc[:,3:])

    def _compute_finger_data_arrays(self):
        """
        Helper method to compute finger data arrays.
        Returns tuple of (pos, req, pos_prev, req_prev) arrays.
        """
        diffs = ["finger_a", "finger_b", "finger_c", "scissor"]
        finger_topics = [f"data/gripper_status_{d}_status" for d in diffs]

        # Each entry is a list where entry[t] is something like [req, pos]
        # Stack to arrays of shape (T, 2) per finger.
        series = [np.asarray(self.proc_dataset[topic]) for topic in finger_topics]  # 4 × (T, 2)

        # Columns: 0=request, 1=position
        # Build (T, 4) matrices by column-stacking across fingers.
        pos = np.column_stack([s[:, 1] for s in series])  # (T, 4)
        req = np.column_stack([s[:, 0] for s in series])  # (T, 4)

        # Previous values: shift down by 1; first row equals current (your rule for i==0).
        pos_prev = pos.copy()
        pos_prev[1:] = pos[:-1]
        # pos_prev[0] already equals pos[0] by copy(); same effect as your offset logic.

        req_prev = req.copy()
        req_prev[1:] = req[:-1]
        
        return pos, req, pos_prev, req_prev

    def _add_finger_specifics(self):
        """
        Vectorized: build everything at once
        """
        pos, req, pos_prev, req_prev = self._compute_finger_data_arrays()

        # Write back as lists of per-timestep vectors to match SmartDict.add semantics.
        # If you prefer to keep them as arrays, you can also just assign the arrays directly.
        self.proc_dataset["data/gripper_status_position"] = [row for row in pos]
        self.proc_dataset["data/gripper_status_position_request"] = [row for row in req]
        self.proc_dataset["data/gripper_status_position_previous"] = [row for row in pos_prev]
        self.proc_dataset["data/gripper_status_position_request_previous"] = [row for row in req_prev]

    def _add_wrench(self):
        """
        Add wrench history windows: for each processed timestamp, 
        find the corresponding position in raw wrench data and take the last N values as history.
        Pad left with the initial value if there aren't enough historical values.
        """
        T = len(self.timestamps)
        if T == 0:
            return

        N = int(self.wrench_hist_window)  # Number of historical wrench values to include

        # N = history length, D = feature dim, T = #processed timestamps, M = #wrench samples
        for wrench_topic in self.wrench_topics:
            wrench_ts_arr = np.asarray(self.raw_dataset[f"{wrench_topic}/timestamp"])   # (M,)
            wrench_data   = np.asarray(self.raw_dataset[f"{wrench_topic}/data"])        # (M, D)
            s_key = f"data/{wrench_topic.split('/')[-1]}_data_window"

            # Indices of the first sample >= ts (rightmost insert pos) — includes sample at ts if equal
            end_idx = np.searchsorted(wrench_ts_arr, self.timestamps, side='right')     # (T,)

            # Front-pad with N copies of the first sample so early timestamps auto-pad
            pad      = np.repeat(wrench_data[[0]], N, axis=0)                           # (N, D)
            data_pad = np.vstack([pad, wrench_data])                                    # (M+N, D)

            # Build all sliding windows once: shape (M+1, N, D)
            # windows_all[i] == last N rows ending at original index i (exclusive)
            windows_all = np.lib.stride_tricks.sliding_window_view(
                data_pad, window_shape=(N,), axis=0
            )                                                                            # (M+1, D, N)

            # Gather the window for each processed timestamp: shape (T, D, N)
            wrench_windows = windows_all[end_idx]

            # If your downstream expects a list of (N, D) arrays per timestep:
            self.proc_dataset[s_key] = [w.T for w in wrench_windows]
    
    def _add_progress(self):
        """Auxiliary learning progress prediction"""
        if not self.use_aux_learn:
            return
        joint_angles = np.stack(self.proc_dataset["data/joint_angles_data"][:])
        gripper_request = np.stack(self.proc_dataset["data/gripper_status_position_request"])[:,0]
        progress = np.linspace(0.0, 1.0, len(self.timestamps))                # 1
        
        # Stack together
        for i, ts in enumerate(self.timestamps):
            self.proc_dataset.add("data/robot_act_aux", np.hstack([                      # TOTAL dim 9
                joint_angles[i], gripper_request[i], progress[i]
            ]))

    def _add_extra_fields(self):
        """
        Add any extra fields to the processed dataset as needed.
        Vectorized end-to-end with proper indexing.
        """
        # --- Pull arrays once ---
        q        = np.stack(self.proc_dataset["data/joint_angles_data"])                  # (N, 7)
        g_req    = np.stack(self.proc_dataset["data/gripper_status_position_request"])[:,0:1]    # (N,) or (N,1)
        # ensure column shape for hstack
        g_req_c  = g_req.reshape(-1, 1)                                                         # (N, 1)

        ee_tgt_t = np.stack(self.proc_dataset["pose/target_victor_right_tool0_translation"])  # (N, 3)
        ee_tgt_r = np.stack(self.proc_dataset["pose/target_victor_right_tool0_rotation"])     # (N, 4)

        q_meas   = np.stack(self.proc_dataset["data/motion_status_measured_joint_position"])  # (N, 7)
        g_meas   = np.stack(self.proc_dataset["data/gripper_status_position"])                # (N, 4)

        ee_cur_t = np.stack(self.proc_dataset["pose/victor_right_tool0_translation"])         # (N, 3)
        ee_cur_r = np.stack(self.proc_dataset["pose/victor_right_tool0_rotation"])            # (N, 4)

        # --- Vectorized action stacks ---
        robot_act    = np.hstack([q, g_req_c])                            # (N, 8)
        robot_act_ee = np.hstack([ee_tgt_t, ee_tgt_r, g_req_c])           # (N, 8)

        # --- "Previous action" per timestep (i=0 uses last) ---
        prev_robot_act    = np.roll(robot_act,    shift=1, axis=0)        # (N, 8)
        prev_robot_act_ee = np.roll(robot_act_ee, shift=1, axis=0)        # (N, 8)

        diff_robot_act = robot_act - prev_robot_act
        prev_diff_robot_act = np.roll(diff_robot_act, shift=1, axis=0)
        diff_robot_act_ee = robot_act_ee - prev_robot_act_ee
        prev_diff_robot_act_ee = np.roll(diff_robot_act_ee, shift=1, axis=0)

        # --- Observations (current measurements + previous action) ---
        robot_obs    = np.hstack([prev_robot_act,    q_meas, g_meas])     # (N, 8+7+4=19)
        diff_robot_obs = np.hstack([prev_diff_robot_act,    q_meas, g_meas]) # (N, 8+7+4=19)
        robot_obs_ee = np.hstack([prev_robot_act_ee, ee_cur_t, ee_cur_r, g_meas])  # (N, 8+3+4+4=19)
        diff_robot_obs_ee = np.hstack([prev_diff_robot_act_ee, ee_cur_t, ee_cur_r, g_meas])  # (N, 8+3+4+4=19)

        # Even though compute is parallelized, adding is still sequential to maintain shapes
        for i, ts in enumerate(self.timestamps):
            # joint space act
            self.proc_dataset.add("data/robot_act", robot_act[i])
            self.proc_dataset.add("data/robot_act_diff", diff_robot_act[i])
            # ee space act
            self.proc_dataset.add("data/robot_act_ee", robot_act_ee[i])
            self.proc_dataset.add("data/robot_act_diff", diff_robot_act[i])

            # joint space obs
            self.proc_dataset.add("data/robot_obs", robot_obs[i])
            self.proc_dataset.add("data/robot_obs_diff", diff_robot_obs[i])
            # ee space obs
            self.proc_dataset.add("data/robot_obs_ee", robot_obs_ee[i])
            self.proc_dataset.add("data/robot_obs_ee_diff", diff_robot_obs_ee[i])

    def _add_episode_ends(self, ep):
        """As titled"""
        if ep in self.train_epi:
            self.proc_dataset.add("meta/episode_ends",self.last_train_ep_end + len(self.proc_dataset["data/timestamp"]))
            self.last_train_ep_end += len(self.proc_dataset["data/timestamp"])
        else:
            self.proc_dataset.add("meta/episode_ends",self.last_val_ep_end + len(self.proc_dataset["data/timestamp"]))
            self.last_val_ep_end += len(self.proc_dataset["data/timestamp"])
        self.proc_dataset.add("meta/episode_name", str(ep))

        # if self.use_aux_learn: # normalize the last dimension for progress tracking
        #     print(np.array(self.proc_dataset["data/robot_act"])[..., -1], i, len(self.timestamps))
        #     self.proc_dataset["data/robot_act"][..., -1] = self.proc_dataset["data/robot_act"][..., -1] / i
        #     self.proc_dataset["data/robot_act_ee"][..., -1] = self.proc_dataset["data/robot_act_ee"][..., -1] / i

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "A post-processor script to take the raw dataset and create a processed dataset for training a Diffusion Policy model")
    parser.add_argument("-c", "--config", help = "path to the config file", required = True)
    argument = parser.parse_args()

    proc_config = BagProcessorConfig()

    config_path = argument.config
    if not os.path.exists(config_path) or not config_path.endswith(".yaml"):
        raise ValueError("Invalid config file: {}".format(config_path))
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    proc_config = proc_config.update(config)
    bag_proc = BagPostProcessor(
        config=proc_config
    )
    bag_proc.post_process()
