import zarr
import zarr.errors
import zarr.hierarchy
import zarr.storage
import numpy as np
import matplotlib.pyplot as plt

import h5py
from utils.imagecodecs_numcodecs import register_codecs
register_codecs()

def print_zarr_vals(obj):
    if isinstance(obj, zarr.hierarchy.Group):
        return # do not print the groups
    # print(obj.basename)
    print(obj.name)
    print(np.array(obj))

def print_zarr_keys(name):
    print(name)

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=5, threshold=77777777)
    # dirf = zarr.open("baselines/diffusion_policy/pusht_cchi_v7_replay.zarr.zip", mode='r')
    # dirf = zarr.open("data/victor/tmp//dspro_07_22_no_wrench.zarr.zip", mode='r')
    # zf = zarr.open("data/victor/tmp//dspro_07_28_end_affector.zarr.zip", mode='r')
    zf = zarr.open("/home/KirillT/robot_tool_use_diffusion_policy/data/victor/victor_playback_data.zarr.zip", mode='r')
    print(zf.tree())    # view all topics and their hierarchy
    
    # ...existing code...
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    pos = zf["robot_act_ea"][:,:3]
    # pos = zf["pose/target_victor_right_tool0/translation"]
    colors = np.linspace(1, 0, len(pos))

    # Calculate the ranges for each axis
    x_range = pos[:, 0].max() - pos[:, 0].min()
    y_range = pos[:, 1].max() - pos[:, 1].min()
    z_range = pos[:, 2].max() - pos[:, 2].min()
    max_range = max(x_range, y_range, z_range)

    # Calculate the centers
    x_mid = (pos[:, 0].max() + pos[:, 0].min()) * 0.5
    y_mid = (pos[:, 1].max() + pos[:, 1].min()) * 0.5
    z_mid = (pos[:, 2].max() + pos[:, 2].min()) * 0.5
    # 20718123604
    # 19329252572
    # 21389443751
    # Set equal limits around the center
    ax.set_xlim(x_mid - max_range*0.5, x_mid + max_range*0.5)
    ax.set_ylim(y_mid - max_range*0.5, y_mid + max_range*0.5)
    ax.set_zlim(z_mid - max_range*0.5, z_mid + max_range*0.5)

    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=colors, cmap='winter')
    ax.set_box_aspect([1, 1, 1])  # Set equal aspect ratio
    # ...existing code...
    # ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], cmap="iridis")
    plt.title("recreated path (sample prediction type)")
    # plt.title("original path")
    plt.show()
   
