from victor_hardware_interfaces.msg import MotionStatus
from victor_hardware_interfaces.msg import CartesianControlModeLimits  # noqa: F401
from victor_hardware_interfaces.msg import CartesianImpedanceParameters  # noqa: F401
from victor_hardware_interfaces.msg import CartesianPathExecutionParameters  # noqa: F401
from victor_hardware_interfaces.msg import CartesianValueQuantity  # noqa: F401
from victor_hardware_interfaces.msg import ControlMode  # noqa: F401
from victor_hardware_interfaces.msg import ControlModeParameters  # noqa: F401
from victor_hardware_interfaces.msg import GraspStatus  # noqa: F401
from victor_hardware_interfaces.msg import JointImpedanceParameters  # noqa: F401
from victor_hardware_interfaces.msg import JointPathExecutionParameters  # noqa: F401
from victor_hardware_interfaces.msg import JointValueQuantity  # noqa: F401
from victor_hardware_interfaces.msg import MotionStatus  # noqa: F401
from victor_hardware_interfaces.msg import Robotiq3FingerActuatorCommand  # noqa: F401
from victor_hardware_interfaces.msg import Robotiq3FingerActuatorStatus  # noqa: F401
from victor_hardware_interfaces.msg import Robotiq3FingerCommand  # noqa: F401
from victor_hardware_interfaces.msg import Robotiq3FingerObjectStatus  # noqa: F401
from victor_hardware_interfaces.msg import Robotiq3FingerStatus  # noqa: F401
from geometry_msgs.msg import WrenchStamped
# from utils.msg import WrenchStamped
from sensor_msgs.msg import Image

def test():
    print(dir(MotionStatus))
    print(dir(WrenchStamped))
    test_msg = MotionStatus()
    test_msg.header.frame_id = "test"
    print(test_msg)

if __name__ == "__main__":
    test()
