import argparse
import os

from rclpy.time import Time
from tf2_ros.buffer import Buffer
import tf2_py as tf2

import numpy as np

from std_msgs.msg import String
# TODO NOTE: every time you want to run this script, make sure to 
#               source ros/install/setup.bash
#            (after building victor_hardware_interfaces of course)
# from victor_hardware_interfaces.msg import MotionStatus, Robotiq3FingerStatus

from .ros_utils import *
from .data_utils import (
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
    def __init__(self, side, vis, aux, inter, hz = 10, wrench_ratio = 10):
        self.hz = hz
        self.wrench_ratio = wrench_ratio    # number of wrench messages per window
        self.freq = 1/hz # NOTE: timestamps are in ns   # TODO no longer needed due to the zivid times

        self.use_vision = vis # should the processed ds include images or not
        self.use_aux_learn = aux
        self.use_interpolation = inter
        self.tracked_sides = [side]
        # persistent processed dataset
        self.pro_dataset = SmartDict(backend="numpy")  # processed dataset 
        self.last_ep_end = 0

    # sets everything up to process a new raw file
    def __setup(self, zarr_dir, image_dir):
        self.zarr_dir = zarr_dir
        self.img_dir = image_dir

        # forcing the buffer to keep all the updates (up to an hr)
        self.buffer = Buffer(cache_time=Duration(seconds=3600))

        zf = zarr.open(self.zarr_dir, mode='r')
        # print(zf.tree())
        self.raw_dataset = SmartDict(backend="numpy")
        read_zarr_dict_recursive(zf, self.raw_dataset)
        # print(self.raw_dataset)


        ### SETUP TRACKED TOPICS
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
                # a = get_h5_leaf_topics(hf, "/left_arm/motion_status", 0, 1)
                # a.remove("timestamp")
                if pt in manual_parent_topics:  continue

                key = "/" + side + "_arm/" + pt
                if key + "/timestamp" not in self.raw_dataset.keys():   continue    # no data was recorded for this topic
                self.tracked_topics[key] = [t for t in zf[key].keys() if "timestamp" not in t]
                # TODO removed the per arm tracking
                # self.tracked_topics[pt] = [t for t in get_h5_leaf_topics(hf, key, 0, 1) if "timestamp" not in t]
                # self.tracked_topics.extend()

            self.tracked_topics["/" + side + "_arm/gripper_status"] = ["finger_a_status", "finger_b_status", "finger_c_status", "scissor_status"]
            # measured joint position is used for the obs, while joint_angles are the commands
            self.tracked_topics["/" + side + "_arm/motion_status"] = ["measured_joint_position"] #, "commanded_joint_position"]

        # manually adding topics that do not have a left and a right side
        # print(self.tracked_topics)

        # contains all the wrench topics that should be tracked at a higher frequency
        self.wrench_topics = ["/" + side + "_arm/wrench"]   # NOTE ASSUMES THAT THIS TOPIC HAS /data

        # contains all the transform frames in the dataset,
        #  not necessarily all that will be included in the final dataset
        self.tf_buffer_topics = []
        for side in self.tracked_sides:
            self.tf_buffer_topics.extend([f"/{side}_arm/pose/" + k for k in zf[f"/{side}_arm/pose"].keys()])
            self.tf_buffer_topics.extend([f"/{side}_arm/pose_static/" + k for k in zf[f"/{side}_arm/pose_static"].keys()])
        # print("tf_buffer:", self.tf_buffer_topics)
        self.tf_buffer_reference_topics = ["/reference_pose/" + k for k in zf["/reference_pose"].keys()]
        # print(self.tf_buffer_reference_topics)

        ### SETUP timestamps
        evt = self.__find_earliest_valid_time()
        # self.timestamps = list(range(evt, int(self.raw_dataset["/duration"][0]), int(self.freq * 1e9)))

        if self.use_vision:
            evt_zivid = np.searchsorted(self.raw_dataset["/zivid/timestamp"], evt)
            self.timestamps = self.raw_dataset["/zivid/timestamp"][evt_zivid:]
            # TODO WRONG
            # self.wrench_timestamps = self.raw_dataset["/wrench/timestamp"][evt_zivid:]
            self.img_h5 = None
        else:
            self.timestamps = list(range(evt, int(self.raw_dataset["/duration"][0]), int(self.freq * 1e9)))

    # returns the topics we want to track in the final dataset
    def __get_tf_topics(self, side: String):
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
    
    def post_process(self, data_dir, ep):
        self.__setup(os.path.join(data_dir, ep, "raw", ep + ".zarr.zip"), os.path.join(data_dir, ep, "zivid"))
        print("processing", self.zarr_dir,"...")
        self.__first_pass()
        self.__second_pass(ep)

    # add a tf topic to the tf buffer
    def __process_tf_topic(self, key):
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
    def __first_pass(self):
        print("\n--------------------------------------------------------------\nperforming the first pass...")
        # print(self.raw_dataset)
        for k in self.tf_buffer_topics:
            print(f"tf topic: {k}")
            self.__process_tf_topic(k)

        for k in self.tf_buffer_reference_topics:
            print("reference topic:", k)
            self.__process_tf_topic(k)
            # print("here")
        print("first pass COMPLETE! \n--------------------------------------------------------------\n")


    # finds the earliest time when at all tracked topics have published at least 1 msg
    def __find_earliest_valid_time(self):
        earliest_ts = 0   

        for side in self.tracked_sides:
            tf_topics = self.__get_tf_topics(side)
            # TODO: ugly way of doing this
            for t in self.parent_topics: 
                k = "/" + side + "_arm/" + t + "/timestamp"
                if k not in self.raw_dataset.keys():    continue
                if self.raw_dataset[k][0] > earliest_ts:
                    earliest_ts = self.raw_dataset[k][0]
                    # print(earliest_ts)
            for t in tf_topics:
                k = "/" + side + "_arm/pose/" + t + "/timestamp"
                if k not in self.raw_dataset.keys():    continue
                if self.raw_dataset[k][0] > earliest_ts:
                    earliest_ts = self.raw_dataset[k][0]
                    # print(earliest_ts)
            for t in tf_topics:
                k = "/" + side + "_arm/pose_static/" + t + "/timestamp"
                if k not in self.raw_dataset.keys():    continue
                if self.raw_dataset[k][0] > earliest_ts:
                    earliest_ts = self.raw_dataset[k][0]
                    # print(earliest_ts)
        return earliest_ts
    
    # finds the eqn of a line from 2 points and returns the value @ x
    def __point_slope_fit(self, x_1, y_1, x_2, y_2, x):
        m = 0
        if x_2 != x_1:
            m = (y_2 - y_1) / (x_2 - x_1)
        return m * (x - x_1) + y_1

    # go through all topics and calculate the values
    def __second_pass(self, ep):
        print("\n--------------------------------------------------------------\nperforming the second pass...")

        # TODO vestigial remnants of when we used to track this
        tf_topics = []
        for side in self.tracked_sides:
            tf_topics.extend(self.__get_tf_topics(side))

        print("tf topics:", tf_topics)
        for i, ts in enumerate(self.timestamps):
        #     # print()
            self.pro_dataset.add("data/timestamp", ts)
            #   1. Go through the transformations and use the buffer to look up the state at that time
            for tf_t in tf_topics:
                # print(tf_t)
                try:
                    tr = bag_proc.buffer.lookup_transform("victor_root", tf_t, Time(nanoseconds=ts))
                    # print(tr)
                    self.pro_dataset.add("pose/" + tf_t + "_translation", ros_msg_to_arr(tr.transform.translation))
                    self.pro_dataset.add("pose/" + tf_t + "_rotation", ros_msg_to_arr(tr.transform.rotation))
                # except tf2.LookupException: pass # if the topic has no data, ignore (happens when only one of the arms was used)
                except tf2.ExtrapolationException:  # TODO if the topic has run out of the data, fill the rest with the last known value
                    tr = bag_proc.buffer.lookup_transform("victor_root", tf_t, Time()) # get latest message
                    self.pro_dataset.add("pose/" + tf_t + "_translation", ros_msg_to_arr(tr.transform.translation))
                    self.pro_dataset.add("pose/" + tf_t + "_rotation", ros_msg_to_arr(tr.transform.rotation))
                    # ALTERNATIVELY do the latest data from the dict
                    # self.pro_dataset.add("pose/" + topic + "/translation", self.pro_dataset["pose/" + topic  + "/translation"][-1])
                    # self.pro_dataset.add("pose/" + topic + "/rotation", self.pro_dataset["pose/" + topic  + "/rotation"][-1])

            #   2. Go through the rest of the variables and do linear approx 
            for topic in self.tracked_topics.keys():
                try:
                    ts_arr = self.raw_dataset[topic + "/timestamp"]
                    # high_i = np.where(ts_arr > ts)
                    # low_i = ts_arr[ts_arr < ts].max()
                    low_i = np.searchsorted(ts_arr, ts, "right") - 1
                    high_i = np.searchsorted(ts_arr, ts, "left")
                    # print(ts)
                    # print(low_i, high_i, sep="\t")
                    # print(ts_arr[low_i], ts,  ts_arr[high_i], sep="\t")
                    # print(ts_arr[high_i] - ts_arr[low_i])

                    # TODO if we are now extrapolating
                    if high_i >= len(ts_arr):
                        for value_topic in self.tracked_topics[topic]:
                            key = topic + "/" + value_topic 
                            # key=value_topic
                            # self.pro_dataset.add(key, self.pro_dataset[key][-1])  # last timestamp value
                            # TODO removed per arm directories for data
                            # self.pro_dataset.add(key, self.raw_dataset[key][-1])    # last known value
                            s_key = "data/" + topic.split("/")[-1] + "_" + value_topic
                            # print(s_key)
                            # self.pro_dataset.add(key, self.raw_dataset[key][-1])    # last known value
                            self.pro_dataset.add(s_key, self.raw_dataset[key][-1])    # last known value
                            # self.pro_dataset.add(key, np.full(self.pro_dataset[key][-1].shape, np.nan)) # NaNs
                    else:
                        for value_topic in self.tracked_topics[topic]:
                            key = topic + "/" + value_topic 
                            s_key = "data/" + topic.split("/")[-1] + "_" + value_topic
                            # print(s_key)

                            # to interpolate or not to interpolate that is the question - Kirillpeare 2025
                            if self.use_interpolation:
                                self.pro_dataset.add(s_key, self.__point_slope_fit(
                                    x_1 = ts_arr[low_i], y_1 = self.raw_dataset[key][low_i],
                                    x_2 = ts_arr[high_i], y_2 = self.raw_dataset[key][high_i],
                                    x = ts
                                ))
                            else:
                                self.pro_dataset.add(s_key, self.raw_dataset[key][low_i])
                except KeyError:    # if a topic doesn't have data (like when only one arm was used)
                    print("ouch", topic, "missing")
                    pass
            
            if self.use_vision:
                img_i = np.searchsorted(self.raw_dataset["/zivid/timestamp"], ts) 
                fid = self.raw_dataset["/zivid/frame_id"][img_i]
                if self.img_h5 is None or fid % 50 == 0:   # we've finished one chunk -> load next
                    self.img_h5 = h5py.File(os.path.join(self.img_dir, "processed_chunk_" + str(int(fid / 50)) + ".h5"))
                try:
                    self.pro_dataset.add("data/image", self.img_h5["rgb"][fid % 50])
                except IndexError:
                    print("index", fid % 50,"in chunk", int(fid / 50), "is out of bounds, using last image")
                    self.pro_dataset.add("data/image", self.img_h5["rgb"][-1])
            else:
                # self.pro_dataset.add("data/image", np.zeros((2,2,3)))
                pass

            # add dedicated position and position request groups
            self.pro_dataset.add("data/gripper_status_position", np.hstack([
                self.pro_dataset["data/gripper_status_finger_a_status"][-1][1],   # 1
                self.pro_dataset["data/gripper_status_finger_b_status"][-1][1],   # 1
                self.pro_dataset["data/gripper_status_finger_c_status"][-1][1],   # 1
                self.pro_dataset["data/gripper_status_scissor_status"][-1][1],    # 1
            ]))
            
            self.pro_dataset.add("data/gripper_status_position_request", np.hstack([
                self.pro_dataset["data/gripper_status_finger_a_status"][-1][0],   # 1
                self.pro_dataset["data/gripper_status_finger_b_status"][-1][0],   # 1
                self.pro_dataset["data/gripper_status_finger_c_status"][-1][0],   # 1
                self.pro_dataset["data/gripper_status_scissor_status"][-1][0],    # 1
            ]))

            offset = -2
            if i == 0: # if the first timestamp, we don't have a previous timestamp, so set it equal to the current one
                offset = -1

            self.pro_dataset.add("data/gripper_status_position_previous", np.hstack([
                self.pro_dataset["data/gripper_status_finger_a_status"][offset][1],   # 1
                self.pro_dataset["data/gripper_status_finger_b_status"][offset][1],   # 1
                self.pro_dataset["data/gripper_status_finger_c_status"][offset][1],   # 1
                self.pro_dataset["data/gripper_status_scissor_status"][offset][1],    # 1
            ]))
            
            self.pro_dataset.add("data/gripper_status_position_request_previous", np.hstack([
                self.pro_dataset["data/gripper_status_finger_a_status"][offset][0],   # 1
                self.pro_dataset["data/gripper_status_finger_b_status"][offset][0],   # 1
                self.pro_dataset["data/gripper_status_finger_c_status"][offset][0],   # 1
                self.pro_dataset["data/gripper_status_scissor_status"][offset][0],    # 1
            ]))

            #   3. Go through the wrench topics and add them to the dataset
            for wrench_topic in self.wrench_topics:
                key = wrench_topic + "/data"   
                s_key = "data/" + wrench_topic.split("/")[-1] + "_data_window"

                if i == 0: # if the first timestamp, we don't have a wrench window before it, so set it equal to the wrench @ first timestamp
                    self.pro_dataset.add(s_key, np.tile(self.raw_dataset[wrench_topic + "/data"][0], (self.wrench_ratio,1)))
                    continue

                window_ts = np.linspace(self.timestamps[i-1], ts, self.wrench_ratio)
                window_arr = []
                for wts in window_ts:
                    try:
                        ts_arr = self.raw_dataset[wrench_topic + "/timestamp"]
                        
                        low_i = np.searchsorted(ts_arr, wts, "right") - 1
                        high_i = np.searchsorted(ts_arr, wts, "left")

                        # if we are now extrapolating   NOTE: other extrapolation/interpolation methods can be seen in step 2
                        if high_i >= len(ts_arr):
                            window_arr.append(self.raw_dataset[key][-1])    # last known value
                        else:
                            if self.use_interpolation:
                                window_arr.append(self.__point_slope_fit(
                                    x_1 = ts_arr[low_i], y_1 = self.raw_dataset[key][low_i],
                                    x_2 = ts_arr[high_i], y_2 = self.raw_dataset[key][high_i],
                                    x = wts
                                ))
                            else:
                                window_arr.append(self.raw_dataset[key][low_i])
                    except KeyError:    # if a topic doesn't have data (like when only one arm was used)
                        print("ouch", topic, "missing")
                        pass
                
                self.pro_dataset.add(s_key, np.array(window_arr))



            # 4. Concatenate the robot observations and actions
            if self.use_aux_learn:  # if we are trying to get the model to learn auxiliary tasks (like how along the trajectory progress it is)
                self.pro_dataset.add("data/robot_act_aux", np.hstack([                      # TOTAL dim 9
                    self.pro_dataset["data/joint_angles_data"][-1],                     # 7
                    self.pro_dataset["data/gripper_status_position_request"][-1][0],       # 1
                    i / (len(self.timestamps) - 1)                                         # 1 
                ]))

            self.pro_dataset.add("data/robot_act", np.hstack([                      # TOTAL dim 8
                self.pro_dataset["data/joint_angles_data"][-1],                     # 7
                self.pro_dataset["data/gripper_status_position_request"][-1][0],       # 1
            ]))

            self.pro_dataset.add("data/robot_act_ea", np.hstack([                    # total dim 8
                self.pro_dataset["pose/target_victor_right_tool0_translation"][-1], # 3 dim
                self.pro_dataset["pose/target_victor_right_tool0_rotation"][-1],    # 4
                self.pro_dataset["data/gripper_status_position_request"][-1][0],       # 1
            ]))                

            if i == 0: # if the first timestamp, we don't have a robot action before it, so set it equal to the robot action @ first timestamp
                self.pro_dataset.add("data/robot_obs", np.hstack([                      #TOTAL dim: 19
                    self.pro_dataset["data/robot_act"][-1],                             # 8
                    self.pro_dataset["data/motion_status_measured_joint_position"][-1], # 7
                    self.pro_dataset["data/gripper_status_position"][-1],               # 4
                ])) 
                # print(self.pro_dataset["data/robot_obs"][-1].shape)
                self.pro_dataset.add("data/robot_obs_ea", np.hstack([                   # total dim: 19
                    self.pro_dataset["data/robot_act_ea"][-1],                             # 8
                    self.pro_dataset["pose/victor_right_tool0_translation"][-1],        # 3
                    self.pro_dataset["pose/victor_right_tool0_rotation"][-1],           # 4 
                    self.pro_dataset["data/gripper_status_position"][-1],       # 4
                ]))
            else:
                # TODO for robot_obs -> [previous act, curr observaitons]
                self.pro_dataset.add("data/robot_obs", np.hstack([                      #TOTAL dim: 19
                    self.pro_dataset["data/robot_act"][-2],                             # 8
                    self.pro_dataset["data/motion_status_measured_joint_position"][-1], # 7
                    self.pro_dataset["data/gripper_status_position"][-1],               # 4
                ])) 

                self.pro_dataset.add("data/robot_obs_ea", np.hstack([                   # total dim: 19
                    self.pro_dataset["data/robot_act_ea"][-2],                             # 8
                    self.pro_dataset["pose/victor_right_tool0_translation"][-1],        # 3
                    self.pro_dataset["pose/victor_right_tool0_rotation"][-1],           # 4 
                    self.pro_dataset["data/gripper_status_position"][-1],       # 4
                ]))                  

            # print(self.pro_dataset["data/robot_act"][-1].shape)

        self.pro_dataset.add("meta/episode_ends",self.last_ep_end + len(self.pro_dataset["data/timestamp"]))
        self.last_ep_end += len(self.pro_dataset["data/timestamp"])
        self.pro_dataset.add("meta/episode_name", str(ep))

        # if self.use_aux_learn: # normalize the last dimension for progress tracking
        #     print(np.array(self.pro_dataset["data/robot_act"])[..., -1], i, len(self.timestamps))
        #     self.pro_dataset["data/robot_act"][..., -1] = self.pro_dataset["data/robot_act"][..., -1] / i
        #     self.pro_dataset["data/robot_act_ea"][..., -1] = self.pro_dataset["data/robot_act_ea"][..., -1] / i

        print("second pass COMPLETE! \n--------------------------------------------------------------\n")

    def __wrench_pass(self):
        print("\n--------------------------------------------------------------\nperforming the wrench pass...")
        # go through the wrench topics and add them to the dataset

        print("wrench pass COMPLETE! \n--------------------------------------------------------------\n")
if __name__ == "__main__":
    # print("postprocessing ......")

    parser = argparse.ArgumentParser(description = "A post-processor script to take the raw dataset and create a processed dataset for training a Diffusion Policy model")
    parser.add_argument("-d", "--data", metavar="PATH_TO_DATA_IN_DIR",help = "path to the raw dataset directory",
                        required = False, default = "data_in")
    parser.add_argument("-p", "--processed", metavar="PATH_TO_PROCESSED_FILE", help = "path to save the processed file at",
                         required = False, default = "datasets/data_out/dspro_08_06_new50_no_interp.zarr.zip")
    parser.add_argument("-s", "--side", help = "which should the dataset include", choices=["left", "right"],
                         required = False, default = "right")
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
    # optional args21728129481 - 21693698127
    # bag_proc = BagPostProcessor("datasets/test_ds/ds_raw432.h5")

    # create the zarr and h5 files
    # zp = zarr.ZipStore(path=processed_path, mode="w")
    # h5f = h5py.File("datasets/data_out/ds_processed.h5", mode='w')

    bag_proc = BagPostProcessor(side, vision == 'true', aux == 'true', inter == 'true')
    ep_i = 0
    for ep_dir in os.listdir(data_in_dir):
        bag_proc.post_process(data_in_dir, ep_dir)
        if ep_i == 0:   # TODO ugly :(
            store_zarr_dict_diff_data(processed_path, bag_proc.pro_dataset)
            store_h5_dict("datasets/data_out/ds_processed.h5", bag_proc.pro_dataset)
        else:
            store_zarr_dict_diff_data_a(processed_path, bag_proc.pro_dataset)
            store_h5_dict_a("datasets/data_out/ds_processed.h5", bag_proc.pro_dataset)        # self.pro_dataset.add("meta/episode_ends", 1)

        bag_proc.pro_dataset = SmartDict(backend="numpy")
        ep_i += 1

    # ds_dir = Path(os.path.join("datasets", "test_ds"))d
    # ds_dir.mkdir(parents=True, exist_ok=True)
    
    print()
    
    print("saving processed dataset dict to", processed_path)
    # store_zarr_dict_diff_data(processed_path, bag_proc.pro_dataset)
    # store_h5_dict("datasets/data_out/ds_processed.h5", bag_proc.pro_dataset)
