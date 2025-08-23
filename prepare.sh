export DP_PROJECT_ROOT=${PWD}

# For running Isaac Sim
export ISAAC_SIM_PATH=$HOME/Applications/isaac-sim-4.5.0

_ROS_WS_REL_PATH="$DP_PROJECT_ROOT/diffusion_policy/real_world/ros-ws"
# If ros-ws setup, source that as well
if [ -f "$_ROS_WS_REL_PATH/prepare.sh" ]; then
    source "$_ROS_WS_REL_PATH/prepare.sh"
fi