import zarr
import zarr.errors
import zarr.hierarchy
import zarr.storage
import numpy as np
import matplotlib.pyplot as plt
import argparse

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
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
    parser = argparse.ArgumentParser(description="Visualize and inspect zarr.zip files")
    parser.add_argument("-f", "--file", metavar="PATH_TO_ZARR_FILE", 
                       help="path to the zarr.zip file to inspect", required=True)
    args = parser.parse_args()

    np.set_printoptions(suppress=True, precision=5, threshold=77777777)
    dirf = zarr.open(args.file, mode='r') 
    print(dirf.tree())    # view all topics and their hierarchy
    # print(np.array(dirf["data/robot_obs"][20:100]))
    # print(dirf.visitvalues(print_zarr_vals))
    # print(dirf["right_arm"].tree(False, 2))
    # replace left_arm/pose with any other topic to view its contents
    # print(dirf["left_arm/pose"].tree())
    # print(dirf["left_arm/pose"].visitvalues(print_zarr_vals))
    # print(dirf["left_arm/pose"].visitkeys(print_zarr_keys))
    # print(np.array(dirf["right_arm/gripper_status/scissor_status"]))
    print(np.array(dirf["meta/episode_ends"]))
    print(np.array(dirf["meta/episode_name"]))
    plt.plot(np.array(dirf["data/robot_act"][:,0]))
    plt.show()
    # print()
    # plt.imshow(np.moveaxis(dirf["data/img"][612], -1, 1)/255)
    # print(dirf["data/image"].shape)
    # print(np.array(dirf["data/image"]))
    # plt.imshow(np.array(dirf["data/image"][500]))
    # plt.show()
    

    # hf = h5py.File("rosbag/0618_images/0618_traj1/zivid2_Settings_Zivid_Two_M70_ParcelsMatte_10Hz_4xsparse_enginetop_boxed/processed_chunk_6.h5", "r")
    # plt.imshow(hf["rgb"][12])
    # plt.show()
