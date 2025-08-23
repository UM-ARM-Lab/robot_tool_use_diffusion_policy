"""
I'm sure we'll need some ROS utilities here in the future. 
"""
from rclpy.time import Time, Duration
# from rosbags.typesys import Stores, get_types_from_msg, get_typestore
# from rosbags.rosbag2 import Reader
import numpy as np
from geometry_msgs.msg import PointStamped, TransformStamped, Transform

# subtracts the current timestamp from the timestamp recorded at initialization
def ros_abs_time_to_elapsed_duration(t: Time, init_t: Time) -> Duration:
    return t - init_t
    
# returns the miliseconds of the rclpy.time.Time timestamp
def ros_duration_to_ns(d: Duration) -> np.int64:
    return d.nanoseconds

# returns the miliseconds of the rclpy.time.Time timestamp
def ros_time_to_ms_float(t: Time) -> np.float64:
    return t.seconds_nanoseconds()[0] * 1e3 + t.seconds_nanoseconds()[1] / 1e6

# converts a ROS message with individual float fields into an array of float64s 
# make sure the passed in message only has numerical fields
# can also include a header field, which will be ignored
def ros_msg_to_arr(msg) -> np.ndarray[np.float64]:
    f_dict = msg.get_fields_and_field_types()
    if "header" in f_dict.keys():   # adds support for types like Robotiq3FingerActuatorStatus, etc
        # return ros_message_to_array_ignore_header(msg)
        f_dict.pop("header")

    arr = []
    for field in f_dict.keys():
        arr.append(msg.__getattribute__(field))
    return np.array(arr, np.float64)    # TODO: currently setting it to be a float

def np_arrs_to_tf_transform(frame_id, child_frame_id, stamp, translation, rotation):
    t = TransformStamped()
    t.header.frame_id = frame_id
    t.header.stamp = Time(nanoseconds=stamp).to_msg()

    t.child_frame_id = child_frame_id
    
    t.transform.translation.x = translation[0]
    t.transform.translation.y = translation[1]
    t.transform.translation.z = translation[2]
    t.transform.rotation.x = rotation[0]
    t.transform.rotation.y = rotation[1]
    t.transform.rotation.z = rotation[2]
    t.transform.rotation.w = rotation[3]

    return t

### ROSBAGS utilities
#ROS Time object constructed from a rosbags Time object
def rosbags_time_to_ros_time(t):
    return Time(seconds = t.sec, nanoseconds = t.nanosec)

def rosbags_msg_to_arr(msg) -> np.ndarray[np.float64]:
    # print(dir(msg))
    f_dict = msg.__dict__
    f_dict.pop("__msgtype__")
    if "header" in f_dict.keys():   # adds support for types like Robotiq3FingerActuatorStatus, etc
        # return ros_message_to_array_ignore_header(msg)
        f_dict.pop("header")

    arr = []
    for field in f_dict.keys():
        arr.append(f_dict[field])
    return np.array(arr, np.float64)    # TODO: currently setting it to be a float

def rosbags_tf_msg_to_transform_obj(msg) -> TransformStamped:
    # print(msg)
    # print(dir(msg))
    t = TransformStamped()
    t.header.frame_id = msg.header.frame_id
    t.header.stamp = rosbags_time_to_ros_time(msg.header.stamp).to_msg()

    t.child_frame_id = msg.child_frame_id
    
    t.transform.translation.x = msg.transform.translation.x
    t.transform.translation.y = msg.transform.translation.y
    t.transform.translation.z = msg.transform.translation.z
    t.transform.rotation.x = msg.transform.rotation.x
    t.transform.rotation.y = msg.transform.rotation.y
    t.transform.rotation.z = msg.transform.rotation.z
    t.transform.rotation.w = msg.transform.rotation.w

    return t

def rosbags_tf_msg_to_transform_obj_custom_time(msg, time) -> TransformStamped:
    # print(msg)
    # print(dir(msg))
    t = TransformStamped()
    t.header.frame_id = msg.header.frame_id
    # t.header.stamp = rosbags_time_to_ros_time(msg.header.stamp).to_msg()
    t.header.stamp = time.to_msg()

    t.child_frame_id = msg.child_frame_id
    
    t.transform.translation.x = msg.transform.translation.x
    t.transform.translation.y = msg.transform.translation.y
    t.transform.translation.z = msg.transform.translation.z
    t.transform.rotation.x = msg.transform.rotation.x
    t.transform.rotation.y = msg.transform.rotation.y
    t.transform.rotation.z = msg.transform.rotation.z
    t.transform.rotation.w = msg.transform.rotation.w

    return t
