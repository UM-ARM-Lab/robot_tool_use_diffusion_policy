import argparse
from pathlib import Path
import os
import json

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

from rclpy.time import Time
import numpy as np

from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import WrenchStamped
# TODO NOTE: every time you want to run this script, make sure to 
#               source ros/install/setup.bash
#            (after building victor_hardware_interfaces of course)
# from victor_hardware_interfaces.msg import MotionStatus, Robotiq3FingerStatus

from .ros_utils import *
from .data_utils import store_h5_dict, SmartDict, store_zarr_dict


def guess_msgtype(path: Path) -> str:
    """Guess message type name from path."""
    name = path.relative_to(path.parents[2]).with_suffix('')
    if 'msg' not in name.parts:
        name = name.parent / 'msg' / name.name
    return str(name)

class BagProcessor():
    def __init__(self, m_d, s):
        self.split = s == "true"
        self.msg_dir = Path(m_d)
        # print(self.msg_dir)
        self.msg_file_names = os.listdir(self.msg_dir)    # assumes only .msg files in dir
        
        # load the default typestore + add custom victor_hardware_interfaces msgs
        self.typestore = get_typestore(Stores.ROS2_HUMBLE)
        add_types = {}

        for msg_name in self.msg_file_names:
            if ".msg" not in msg_name:  continue
            msgpath = Path.joinpath(self.msg_dir, msg_name)
            msgdef = msgpath.read_text(encoding='utf-8')
            add_types.update(get_types_from_msg(msgdef, guess_msgtype(msgpath)))

        self.typestore.register(add_types)

        # contains all the topics tracked and what method should be used to process_ them
        self.tracked_topics = {
            "/victor/left_arm/wrench" : self.__process_wrench,
            "/victor/right_arm/wrench" : self.__process_wrench,

            '/left_arm_impedance_controller/commands' : self.__process_commands,
            '/right_arm_impedance_controller/commands' : self.__process_commands,

            '/victor/left_arm/motion_status' : self.__process_motion_status,
            '/victor/right_arm/motion_status' : self.__process_motion_status,

            '/victor/left_arm/gripper_status' : self.__process_gripper_status,
            '/victor/right_arm/gripper_status' : self.__process_gripper_status,

            "/tf_static" : self.__process_tf,
            "/tf" : self.__process_tf,

            "/zivid_node_local/frame_id" : self.__process_zivid
        }

        self.tf_ignored_frames = {
            "HTC Vive Controller OpenXR right",
            "HTC Vive Controller OpenXR left",
            "controller_right_arm_in_vr0",
            "controller_right_arm_in_vr",
            "controller_left_arm_in_vr",
            "controller_left_arm_in_vr0",
            "vr"
        }

        # init a new SmartDict 
        self.reset()

    def reset(self):
        self.dataset = SmartDict(backend="numpy")

    def elapsed_ns(self, t: Time) -> np.int64:    # more convenient wrapper for the helper methods
        return ros_duration_to_ns(ros_abs_time_to_elapsed_duration(t, self.init_time))

    def process(self, ep_dir, name):
        bag_dir = os.path.join(ep_dir, "rosbag")
        save_dir = os.path.join(ep_dir, "raw")
        annotation_path = os.path.join(ep_dir, "annotation.json")
        split_second = None
        # split
        if self.split:
            split_second = json.load(open(annotation_path, "r"))["keyframes"]['align_cap']

        print("beginning processing", bag_dir, "...")

        with Reader(bag_dir) as reader:
            #initial time is the time the first mesage was received
            self.init_time = Time(nanoseconds=reader.start_time)
            final_second = reader.end_time
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic not in self.tracked_topics.keys():  continue
                # TODO this is for splitting the dataset
                if split_second is not None and self.elapsed_ns(Time(nanoseconds=timestamp)) > split_second * 1e9: 
                    print("splitting at:", split_second, self.elapsed_ns(Time(nanoseconds=timestamp))) 
                    break # 71385413766

                # print('timestamp:', timestamp)

                msg = self.typestore.deserialize_cdr(rawdata, connection.msgtype)

                # print()
                # print(connection.topic)
                if "tf" in connection.topic:
                    self.tracked_topics[connection.topic](msg, Time(nanoseconds=timestamp), "static" in connection.topic)
                elif "zivid" in connection.topic:
                    self.tracked_topics[connection.topic](msg, Time(nanoseconds=timestamp))
                elif "right" in connection.topic:
                    self.tracked_topics[connection.topic](msg, Time(nanoseconds=timestamp), "right")
                else:
                    self.tracked_topics[connection.topic](msg, Time(nanoseconds=timestamp), "left")

                final_second = timestamp

            self.dataset.add("duration", final_second - reader.start_time)

        # print(topics)
        print("finished processing", bag_dir, "bag...")

        # save the dict
        self.save_raw_dict(save_dir, name)
    
        self.reset()    # prepare for the next bag
        print()

    def save_raw_dict(self, save_dir, name):
        ds_dir = Path(save_dir)
        ds_dir.mkdir(parents=True, exist_ok=True)

        print("saving dataset dict to", save_dir)
        store_h5_dict(os.path.join(save_dir, name + ".h5"), self.dataset)
        store_zarr_dict(os.path.join(save_dir, name + ".zarr.zip"), self.dataset)

    def __process_zivid(self, msg, r_time):
        assert msg.__msgtype__ == "std_msgs/msg/Header"
        # t = rosbags_time_to_ros_time(msg.header.stamp)
        t = r_time 
        topic_var = "zivid/"
        data = msg.frame_id.split("/")
        self.dataset.add_static(topic_var + "dataset_name", data[-2])
        self.dataset.add(topic_var + "frame_id", int(data[-1]))
        self.dataset.add(topic_var + "timestamp", self.elapsed_ns(t))


    def __process_wrench(self, msg, r_time, side: String):
        # print(msg.__msgtype__)
        assert msg.__msgtype__ == "geometry_msgs/msg/WrenchStamped"
        
        # t = rosbags_time_to_ros_time(msg.header.stamp) # TODO GARBAGE
        t = r_time 

        w = msg.wrench
        force = rosbags_msg_to_arr(w.force)
        torque = rosbags_msg_to_arr(w.torque)

        topic_var = side + "_arm/wrench/"
        self.dataset.add(topic_var + "data", np.hstack([force, torque]))
        self.dataset.add(topic_var + "timestamp", self.elapsed_ns(t))


    def __process_commands(self, msg, r_time, side: String):
        # print(msg.__msgtype__)
        assert msg.__msgtype__ == "std_msgs/msg/Float64MultiArray"
        # Float64MultiArray does not have a timestamp
        t = r_time 

        topic_var = side + "_arm/joint_angles/"
        self.dataset.add(topic_var +"data", np.array(msg.data))
        self.dataset.add(topic_var + "timestamp", self.elapsed_ns(t))
    

    def __process_motion_status(self, msg, r_time, side: String):
        # print(msg.__msgtype__)
        assert msg.__msgtype__ == "victor_hardware_interfaces/msg/MotionStatus"

        # t = rosbags_time_to_ros_time(msg.header.stamp) # TODO GARBAGE
        t = r_time 

         # could be cleaned up with an attribute loop, but iffy with varying message types/depths
        topic_var = side + "_arm/motion_status/"
        self.dataset.add(topic_var + "measured_joint_position", rosbags_msg_to_arr(msg.measured_joint_position))
        self.dataset.add(topic_var + "commanded_joint_position", rosbags_msg_to_arr(msg.commanded_joint_position))
        self.dataset.add(topic_var + "measured_joint_velocity", rosbags_msg_to_arr(msg.measured_joint_velocity))
        self.dataset.add(topic_var + "measured_joint_torque", rosbags_msg_to_arr(msg.measured_joint_torque))
        self.dataset.add(topic_var + "estimated_external_torque", rosbags_msg_to_arr(msg.estimated_external_torque))
        self.dataset.add(topic_var + "estimated_external_wrench", rosbags_msg_to_arr(msg.estimated_external_wrench))
        self.dataset.add(topic_var + "measured_cartesian_pose_abc", rosbags_msg_to_arr(msg.measured_cartesian_pose_abc))
        self.dataset.add(topic_var + "commanded_cartesian_pose_abc", rosbags_msg_to_arr(msg.commanded_cartesian_pose_abc))
        self.dataset.add(topic_var + "measured_cartesian_pose_position", rosbags_msg_to_arr(msg.measured_cartesian_pose.position))    
        self.dataset.add(topic_var + "measured_cartesian_pose_orientation", rosbags_msg_to_arr(msg.measured_cartesian_pose.orientation))
        self.dataset.add(topic_var + "commanded_cartesian_pose_position", rosbags_msg_to_arr(msg.commanded_cartesian_pose.position))       
        self.dataset.add(topic_var + "commanded_cartesian_pose_orientation", rosbags_msg_to_arr(msg.commanded_cartesian_pose.orientation))      

        self.dataset.add(topic_var + "timestamp", self.elapsed_ns(t))
    

    def __process_gripper_status(self, msg, r_time, side: String):
        # print(msg.__msgtype__)
        assert msg.__msgtype__ == "victor_hardware_interfaces/msg/Robotiq3FingerStatus"

        # t = rosbags_time_to_ros_time(msg.header.stamp) # TODO GARBAGE
        t = r_time 

        self.dataset.add(side + "_arm/gripper_status/finger_a_status", rosbags_msg_to_arr(msg.finger_a_status))
        self.dataset.add(side + "_arm/gripper_status/finger_b_status", rosbags_msg_to_arr(msg.finger_b_status))
        self.dataset.add(side + "_arm/gripper_status/finger_c_status", rosbags_msg_to_arr(msg.finger_c_status))
        self.dataset.add(side + "_arm/gripper_status/scissor_status", rosbags_msg_to_arr(msg.scissor_status))

        self.dataset.add(side + "_arm/gripper_status/timestamp", self.elapsed_ns(t))


    def __process_tf(self, msg, r_time, is_static: bool):
        assert msg.__msgtype__ == "tf2_msgs/msg/TFMessage"
        ti = r_time 
        pose_var = "pose_static" if is_static else "pose"

        for t in msg.transforms:
            if t.child_frame_id in self.tf_ignored_frames:  continue
            # if t.child_frame_id in self.tf_tracked_frames:
            if "right" in t.child_frame_id or "left" in t.child_frame_id:
                side = "right" if "right" in t.child_frame_id else "left"

                topic_var = side + "_arm/" + pose_var + "/" + t.child_frame_id + "/"
                self.dataset.add_static(topic_var + "parent_frame_id", t.header.frame_id)   # TODO 20k strings seem to be causing issues
                self.dataset.add_static(topic_var + "child_frame_id", t.child_frame_id)   # TODO 20k strings seem to be causing issues
                self.dataset.add(topic_var + "translation", rosbags_msg_to_arr(t.transform.translation))
                self.dataset.add(topic_var + "rotation", rosbags_msg_to_arr(t.transform.rotation))
                self.dataset.add(topic_var + "timestamp", self.elapsed_ns(ti))
            # if t.child_frame_id in self.tf_reference_frames:
            else:
                self.dataset.add_static("reference_pose/" + t.child_frame_id + "/parent_frame_id", t.header.frame_id)   # TODO 20k strings seem to be causing issues
                self.dataset.add_static("reference_pose/" + t.child_frame_id + "/child_frame_id", t.child_frame_id)   # TODO 20k strings seem to be causing issues
                self.dataset.add("reference_pose/" + t.child_frame_id + "/translation", rosbags_msg_to_arr(t.transform.translation))
                self.dataset.add("reference_pose/" + t.child_frame_id + "/rotation", rosbags_msg_to_arr(t.transform.rotation))
                self.dataset.add("reference_pose/" + t.child_frame_id + "/timestamp", self.elapsed_ns(ti))

if __name__ == "__main__":
    # print("un-baggin' it")

    parser = argparse.ArgumentParser(description = "A post-processor script to take the raw dataset and create a processed dataset for training a Diffusion Policy model")
    parser.add_argument("-d", "--data", metavar="PATH_TO_DATA_IN_DIR" ,help = "path to the data in directory",
                        required = False, default = "data_in")   #rosbag/rosbag2_2025_06_05-12_29_01
    parser.add_argument("-m", "--msg", metavar="PATH_TO_MSG", help = "path to the victor_hardware_interfaces .msg files",
                         required = False, default = "ros/victor_hardware_interfaces/msg/")
    parser.add_argument('-s', '--split', help = "should the dataset be split before a keyframe", choices=["true", "false"],
                         required = False, default = "true")
    argument = parser.parse_args()

    # required args
    data_dir = argument.data
    # raw_dir = argument.raw
    to_split = argument.split
    
    # optional args
    msg_path = argument.msg    # defaults to ros/victor_hardware_i1terfaces/msg/


    bag_proc = BagProcessor(msg_path, to_split)

    for ep_name in os.listdir(data_dir):
        print(ep_name)
        # bag_proc.process(os.path.join(data_dir, ep_dir, "rosbag"),
        #                  os.path.join(data_dir, ep_dir, "raw"), ep_dir)
        bag_proc.process(os.path.join(data_dir, ep_name), ep_name)

    
    print()
    # move into the process func

