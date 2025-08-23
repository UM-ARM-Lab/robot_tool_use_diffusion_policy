import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .data_utils import store_h5_dict, SmartDict
from .ros_utils import ros_abs_time_to_elapsed_duration, ros_duration_to_ns, ros_msg_to_arr

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import WrenchStamped
# TODO NOTE: every time you want to run this script, make sure to 
#               source ros/install/setup.bash
#            (after building victor_hardware_interfaces of course)
from victor_hardware_interfaces.msg import MotionStatus, Robotiq3FingerStatus


class Subscriber(Node):
    def __init__(self):
        super().__init__('victor_sub')

        # print(self.elapsed_ns(self.get_clock().now()))

        #subscribers
        self.wrench_subscription = self.create_subscription(
            msg_type    = WrenchStamped,
            topic       = '/victor/left_arm/wrench',
            callback    = self.listener_wrench_callback,
            qos_profile = 1
        )
        
        self.action_subscription = self.create_subscription(
            msg_type    = Float64MultiArray,
            topic       = '/left_arm_impedance_controller/commands',
            callback    = self.listener_action_callback,
            qos_profile = 1
        )

        self.motion_status_subscription = self.create_subscription(
            msg_type    = MotionStatus,
            topic       = '/victor/left_arm/motion_status',
            callback    = self.listener_motion_status_callback,
            qos_profile = 1
        )

        self.motion_status_subscription = self.create_subscription(
            msg_type    = Robotiq3FingerStatus,
            topic       = '/victor/left_arm/gripper_status',
            callback    = self.listener_gripper_status_callback,
            qos_profile = 1
        )

        # self.image_subscription = self.create_subscription(
        #     msg_type    = Image,
        #     topic       = '/zivid_node/rgb',
        #     callback    = self.listener_image_callback,
        #     qos_profile = 1
        # )

        self.reset()   # creates empty dict for the dataset

        self.init_time = self.get_clock().now()
        print("INITIAL TIME")
        print(self.init_time)

    def elapsed_ns(self, t: Time) -> np.int64:    # more convenient wrapper for the helper methods
        return ros_duration_to_ns(ros_abs_time_to_elapsed_duration(t, self.init_time))

    def reset(self):
        self.dataset = SmartDict(backend="numpy")

    def listener_wrench_callback(self, msg: WrenchStamped):
        # TODO the current wrench timestamp is runnign a minute ahead, so we'll cheat and use current time
        # t = Time.from_msg(msg.header.stamp)
        t = self.get_clock().now()

        w = msg.wrench
        force = ros_msg_to_arr(w.force)
        torque = ros_msg_to_arr(w.torque)

        self.dataset.add("left_arm/wrench/data", np.hstack([force, torque]))
        self.dataset.add("left_arm/wrench/timestamp", self.elapsed_ns(t))

    def listener_action_callback(self, msg: Float64MultiArray):
        # Float64MultiArray does not have a timestamp, so use Node's time
        t = self.get_clock().now()

        self.dataset.add("left_arm/action/data", np.array(msg.data))
        self.dataset.add("left_arm/action/timestamp", self.elapsed_ns(t))

    def listener_motion_status_callback(self, msg: MotionStatus):
        # t = Time.from_msg(msg.header.stamp)   # TODO running ahead atm
        t = self.get_clock().now()

        # could be cleaned up with an attribute loop, but iffy with varying message types/depths
        self.dataset.add("left_arm/motion_status/measured_joint_position", ros_msg_to_arr(msg.measured_joint_position))
        self.dataset.add("left_arm/motion_status/commanded_joint_position", ros_msg_to_arr(msg.commanded_joint_position))
        self.dataset.add("left_arm/motion_status/measured_joint_velocity", ros_msg_to_arr(msg.measured_joint_velocity))
        self.dataset.add("left_arm/motion_status/measured_joint_torque", ros_msg_to_arr(msg.measured_joint_torque))
        self.dataset.add("left_arm/motion_status/estimated_external_torque", ros_msg_to_arr(msg.estimated_external_torque))
        self.dataset.add("left_arm/motion_status/estimated_external_wrench", ros_msg_to_arr(msg.estimated_external_wrench))
        self.dataset.add("left_arm/motion_status/measured_cartesian_pose_abc", ros_msg_to_arr(msg.measured_cartesian_pose_abc))
        self.dataset.add("left_arm/motion_status/commanded_cartesian_pose_abc", ros_msg_to_arr(msg.commanded_cartesian_pose_abc))
        self.dataset.add("left_arm/motion_status/measured_cartesian_pose/position", ros_msg_to_arr(msg.measured_cartesian_pose.position))    
        self.dataset.add("left_arm/motion_status/measured_cartesian_pose/orientation", ros_msg_to_arr(msg.measured_cartesian_pose.orientation))
        self.dataset.add("left_arm/motion_status/commanded_cartesian_pose/position", ros_msg_to_arr(msg.commanded_cartesian_pose.position))       
        self.dataset.add("left_arm/motion_status/commanded_cartesian_pose/orientation", ros_msg_to_arr(msg.commanded_cartesian_pose.orientation))      

        self.dataset.add("left_arm/motion_status/timestamp", self.elapsed_ns(t))

    def listener_gripper_status_callback(self, msg: Robotiq3FingerStatus):
        # print(ros_message_to_array(msg.finger_a_status))
        # t = Time.from_msg(msg.header.stamp)     # TODO running ahead
        t = self.get_clock().now()

        self.dataset.add("left_arm/gripper_status/finger_a_status", ros_msg_to_arr(msg.finger_a_status))
        self.dataset.add("left_arm/gripper_status/finger_b_status", ros_msg_to_arr(msg.finger_b_status))
        self.dataset.add("left_arm/gripper_status/finger_c_status", ros_msg_to_arr(msg.finger_c_status))
        self.dataset.add("left_arm/gripper_status/scissor_status", ros_msg_to_arr(msg.scissor_status))

        self.dataset.add("left_arm/gripper_status/timestamp", self.elapsed_ns(t))

    def listener_image_callback(self, msg):
        print(msg.header.stamp)
        h = msg.header 
        t = h.stamp
        print("DELAY: ")
        print(self.get_clock().now().seconds_nanoseconds()[1] - t.nanosec)
        # print()

def main(args=None):

    # setup dataset directory
    ds_dir = Path(os.path.join("datasets", "test_ds"))
    ds_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init(args=args) #, signal_handler_options=SignalHandlerOptions.NO)

    # create a data logging ROS node 
    victor_sub = Subscriber()

    try:
        print('data_subscriber launching...')
        rclpy.spin(victor_sub)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        # save the dicts and destroy the node
        print('\n\nshutting down...')

        print("Final dataset:")
        print(victor_sub.dataset)
        # [x[0][0] for x in minimal_subscriber.dataset["left_arm/wrench/data"]]
        seconds = victor_sub.get_clock().now().seconds_nanoseconds()[0]
        print(f"Timestamp = %i" % seconds)
        # store_h5_dict(os.path.join(ds_dir, f"ds_%i.h5" % victor_sub.init_time.seconds_nanoseconds()[0]), victor_sub.dataset)
        store_h5_dict(os.path.join(ds_dir, "ds_raw.h5"), victor_sub.dataset)

        # Visualize wrench data OUTDATED
        # fig, axs = plt.subplots(2,3)
        # plt.suptitle("/left_arm/wrench/data")
        # for row in range(2):
        #     for col in range(3):
        #         axs[row, col].plot([x[row][col] for x in victor_sub.dataset["left_arm/wrench/data"]])
        # plt.tight_layout()
        # plt.show()    

        victor_sub.destroy_node()
        # rclpy.shutdown()        print(minimal_subscriber.dataset)

    
if __name__ == '__main__':
    main()