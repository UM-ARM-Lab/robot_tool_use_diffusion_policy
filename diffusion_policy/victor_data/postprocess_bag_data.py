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

import argparse
import os

from rclpy.time import Time
from tf2_ros.buffer import Buffer
import tf2_py as tf2

import numpy as np
import tqdm

# TODO NOTE: every time you want to run this script, make sure to 
#               source ros/install/setup.bash
#            (after building victor_hardware_interfaces of course)
# from victor_hardware_interfaces.msg import MotionStatus, Robotiq3FingerStatus

from diffusion_policy.victor_data.ros_utils import *
from diffusion_policy.victor_data.data_utils import (
    read_h5_dict_recursive,
    read_zarr_dict_recursive,
    store_h5_dict,
    store_h5_dict_a,
    get_h5_topics,
    get_h5_leaf_topics,
    SmartDict,
    store_zarr_dict,
    store_zarr_dict_diff_data,
    store_zarr_dict_diff_data_a
)

import zarr
import h5py

class BagPostProcessor():
    def __init__(self,
        side,
        vis,
        aux,
        inter,
        hz = 10,
        wrench_ratio = 30
    ):
        self.wrench_ratio = wrench_ratio    # number of wrench messages per window
        self.freq = 1/hz # NOTE: timestamps are in ns   # TODO no longer needed due to the zivid times

        self.use_vision = vis # should the processed ds include images or not
        self.use_aux_learn = aux
        self.use_interpolation = inter
        self.tracked_sides = [side]

        # persistent processed dataset
        self.proc_dataset = SmartDict(backend="numpy")  # processed dataset 
        self.last_ep_end = 0

    def post_process(self, data_dir, ep):
        # Setup
        self.zarr_dir = os.path.join(data_dir, ep, "raw", ep + ".zarr.zip")
        self.img_dir = os.path.join(data_dir, ep, "zivid")

        # forcing the buffer to keep all the updates (up to an hr)
        self.buffer = Buffer(cache_time=Duration(seconds=3600))

        # Read data
        zf = zarr.open(self.zarr_dir, mode='r')
        self.raw_dataset = SmartDict(backend="numpy")
        read_zarr_dict_recursive(zf, self.raw_dataset)

        # Prepare for prcessing
        self._setup_topics(zf)
        self._setup_timestamps()

        # Actual processing
        self._register_transform_buffer()
        self._interpolate_tf()
        self._add_other_values()
        self._add_vision()
        self._add_finger_specifics()
        self._add_wrench()
        self._add_progress()
        self._add_extra_fields()
        self._add_episode_ends(ep)
        # self.__second_pass(ep)

    def _setup_topics(self, zf):
        """
        Setup which topics to track
        """
        # self.tracked_sides = ["left", "right"]
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
        # self.timestamps = list(range(evt, int(self.raw_dataset["/duration"][0]), int(self.freq * 1e9)))

        if self.use_vision:
            evt_zivid = np.searchsorted(self.raw_dataset["/zivid/timestamp"], evt)
            self.timestamps = self.raw_dataset["/zivid/timestamp"][evt_zivid:]
            # TODO WRONG
            # self.wrench_timestamps = self.raw_dataset["/wrench/timestamp"][evt_zivid:]
        else:
            self.timestamps = list(range(evt, int(self.raw_dataset["/duration"][0]), int(self.freq * 1e9)))

        self._filter_timestamps_for_plateau()
        # Add filtered timestamp to dataset
        for ts in self.timestamps:
            self.proc_dataset.add("data/timestamp", ts)

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
        original_count = len(self.timestamps)
        self.timestamps = [self.timestamps[i] for i in range(len(self.timestamps)) if mask[i]]
        filtered_count = len(self.timestamps)
        # print(f"Filtered {original_count - filtered_count} plateau timestamps, kept {filtered_count}")               

    # returns the topics we want to track in the final dataset
    def __get_tf_topics(self, side: str):
        return [
            "target_victor_" + side + "_tool0",
            "victor_" + side + "_tool0",
        
            # "victor_" + side + "_arm_link_0",
            # "victor_" + side + "_arm_link_1",
            # "victor_" + side + "_arm_link_2",
            # "victor_" + side + "_arm_link_3",
            # "victor_" + side + "_arm_link_4",
            # "victor_" + side + "_arm_link_5",
            # "victor_" + side + "_arm_link_6",
            # "victor_" + side + "_arm_link_7",
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
            except IndexError:
                print("index", offset_id,"in chunk", chunk_id, "is out of bounds, using last image")
                self.proc_dataset.add("data/image", img_h5["rgb"][-1])

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
        Vectorized wrench windows: for each interval [t_{i-1}, t_i],
        generate wrench_ratio samples, sampled with "left" (last-known) policy.
        For i==0, window equals wrench at the first timestamp repeated.
        """
        T = len(self.timestamps)
        if T == 0:
            return

        # Build all window targets in one go: shape (T, R)
        t_curr = np.asarray(self.timestamps)
        t_prev = t_curr.copy()
        if T > 1:
            t_prev[1:] = t_curr[:-1]   # previous timestamp per row; t_prev[0] == t_curr[0]

        R = int(self.wrench_ratio)
        # alpha in [0,1] with R points; broadcasting to (T, R)
        alpha = np.linspace(0.0, 1.0, R, dtype=np.float64)
        window_ts = (t_prev[:, None] * (1.0 - alpha)[None, :]) + (t_curr[:, None] * alpha[None, :])

        # Flatten to (T*R,) for a single vectorized sampling per topic
        window_ts_flat = window_ts.reshape(-1)

        for wrench_topic in self.wrench_topics:
            ts_arr = np.asarray(self.raw_dataset[wrench_topic + "/timestamp"])
            y = np.asarray(self.raw_dataset[wrench_topic + "/data"])  # shape (N, D?) or (N,)
            s_key = f"data/{wrench_topic.split('/')[-1]}_data_window"

            # Sample with last-known (left) policy, clamp right to last, left to first
            sampled = self._sample_series(
                ts_arr, y, window_ts_flat,
                mode="left",
                left_policy="first",
                right_policy="last",
            )
            # Reshape back to (T, R, D?) and store as list-of-(R,D) per timestep
            new_shape = (T, R) + y.shape[1:]
            sampled = sampled.reshape(new_shape)

            # If you want the first row equal to "wrench @ first ts" repeated R times,
            # this already holds because t_prev[0] == t_curr[0].
            self.proc_dataset[s_key] = [sampled[i] for i in range(T)]

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

        # --- Observations (current measurements + previous action) ---
        robot_obs    = np.hstack([prev_robot_act,    q_meas, g_meas])     # (N, 8+7+4=19)
        robot_obs_ee = np.hstack([prev_robot_act_ee, ee_cur_t, ee_cur_r, g_meas])  # (N, 8+3+4+4=19)

        for i, ts in enumerate(self.timestamps):
            self.proc_dataset.add("data/robot_act", robot_act[i])
            self.proc_dataset.add("data/robot_act_ee", robot_act_ee[i])
            self.proc_dataset.add("data/robot_obs", robot_obs[i])
            self.proc_dataset.add("data/robot_obs_ee", robot_obs_ee[i])

            
    def _add_episode_ends(self, ep):
        """As titled"""
        self.proc_dataset.add("meta/episode_ends",self.last_ep_end + len(self.proc_dataset["data/timestamp"]))
        self.last_ep_end += len(self.proc_dataset["data/timestamp"])
        self.proc_dataset.add("meta/episode_name", str(ep))

        # if self.use_aux_learn: # normalize the last dimension for progress tracking
        #     print(np.array(self.proc_dataset["data/robot_act"])[..., -1], i, len(self.timestamps))
        #     self.proc_dataset["data/robot_act"][..., -1] = self.proc_dataset["data/robot_act"][..., -1] / i
        #     self.proc_dataset["data/robot_act_ee"][..., -1] = self.proc_dataset["data/robot_act_ee"][..., -1] / i
        

if __name__ == "__main__":
    # print("postprocessing ......")

    parser = argparse.ArgumentParser(description = "A post-processor script to take the raw dataset and create a processed dataset for training a Diffusion Policy model")
    parser.add_argument("-d", "--data", metavar="PATH_TO_DATA_IN_DIR",help = "path to the raw dataset directory",
                        required=True)
    parser.add_argument("-p", "--processed", metavar="PATH_TO_PROCESSED_FILE", help = "path to save the processed file at",
                         required=True)
    parser.add_argument("-s", "--side", help = "which should the dataset include", choices=["left", "right"],
                         required=True)
    parser.add_argument("-v", "--vision", help = "should the dataset use images", choices=["true", "false"],
                         required = False, default = "true")
    parser.add_argument("-a", "--aux", help = "should the model learn auxilary tasks", choices=["true", "false"],
                         required = False, default = "true")    
    parser.add_argument("-i", "--interpolation", help = "should the dataset contain interpolated values", choices=["true", "false"],
                         required = False, default = "true")
    argument = parser.parse_args()

    data_in_dir = argument.data
    processed_path = argument.processed
    # img_dir_path = argument.img #"rosbag/0618_images/0618_traj1/zivid2_Settings_Zivid_Two_M70_ParcelsMatte_10Hz_4xsparse_enginetop_boxed"


    side = argument.side    # defaults to right
    vision = argument.vision
    aux = argument.aux
    inter = argument.interpolation

    bag_proc = BagPostProcessor(side, vision == 'true', aux == 'true', inter == 'true')
    ep_i = 0
    os.makedirs("data/victor/tmp", exist_ok=True)

    pbar = tqdm.tqdm(os.listdir(data_in_dir), desc="Processing episodes")
    for ep_dir in pbar:
        bag_proc.proc_dataset = SmartDict(backend="numpy")
        try:
            bag_proc.post_process(data_in_dir, ep_dir)
        except Exception as e:
            print(f"Error processing {ep_dir}: {e}")
            ep_i += 1
            continue

        # Then save
        if ep_i == 0:   # TODO ugly :(
            store_zarr_dict_diff_data(processed_path, bag_proc.proc_dataset)
            store_h5_dict("data/victor/tmp/ds_processed.h5", bag_proc.proc_dataset)
        else:
            store_zarr_dict_diff_data_a(processed_path, bag_proc.proc_dataset)
            store_h5_dict_a("data/victor/tmp/ds_processed.h5", bag_proc.proc_dataset)
        ep_i += 1

    print("saving processed dataset dict to", processed_path)
