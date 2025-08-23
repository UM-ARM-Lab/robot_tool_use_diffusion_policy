"""
Post-processes .zarr datasets by removing plateau sequences where robot actions remain unchanged.

Inputs:
- Processed .zarr file from postprocess_bag_data.py containing robot trajectories
- Auxiliary learning flag for progress tracking

Outputs:  
- Filtered .zarr file with plateaus removed
- Temporary .h5 file at data/victor/tmp/ds_processed2.h5
"""

import argparse

import numpy as np
import tqdm
# TODO NOTE: every time you want to run this script, make sure to 
#               source ros/install/setup.bash
#            (after building victor_hardware_interfaces of course)
# from victor_hardware_interfaces.msg import MotionStatus, Robotiq3FingerStatus

from diffusion_policy.victor_data.ros_utils import *
from diffusion_policy.victor_data.data_utils import SmartDict

import zarr
import h5py

class ZarrPostProcessor():
    def __init__(self, zf_in_path, aux):
        zs = zarr.ZipStore(path=zf_in_path, mode="r")
        self.zf_in = zarr.open(zs, mode='r')
        self.aux = aux
        self.data_dict = SmartDict()
        self.data_dict["meta/episode_ends"] = []
        # todo handle meta/ep_names since the ends will change
        # todo handle the rest of the data

        # stores all the topics that we want to copy over into the new dataset
        self.topics = ["data/" + k for k in list(self.zf_in["data"].keys())]
        self.topics.extend(["pose/" + k for k in list(self.zf_in["pose"].keys())])
        self.mask = np.ones(self.zf_in["meta/episode_ends"][-1], dtype=int)

    def copy_index(self, i):
        for t in self.topics:
            # print(t)
            self.data_dict.add(t, self.zf_in[t][i])

    # very primitive implementation of this
    def remove_plateaus(self):
        ep_starts_ends = [0]
        ep_starts_ends.extend(self.zf_in["meta/episode_ends"])

        for ep_i in tqdm.tqdm(range(1, len(ep_starts_ends)), "episode #"):
            # self.copy_index(ep_starts_ends[ep_i-1]) 
            ep_len = 0
            # assumes all topics have the same len
            for i in tqdm.tqdm(range(ep_starts_ends[ep_i-1], ep_starts_ends[ep_i]), f"timestamp in ep {ep_i}", leave=False):
                # if all are the same # TODO add a progress value check
                # if self.aux:
                #     # print(np.array(self.zf_in["data/robot_act"][i])[:])
                #     if np.all(np.array(self.zf_in["data/robot_act"][i])[:-1] == np.array(self.zf_in["data/robot_act"][i-1])[:-1]):
                #         self.mask[i] = 0
                #         continue
                # else:
                if np.all(self.zf_in["data/robot_act"][i] == self.zf_in["data/robot_act"][i-1]):
                    self.mask[i] = 0
                    continue                    
                # self.copy_index(i)  # TODO no longer copying anything over at runtime
                ep_len += 1

            # print(ep_len)
            self.data_dict.add("meta/episode_ends", ep_len if len(self.data_dict["meta/episode_ends"]) == 0 else ep_len + self.data_dict["meta/episode_ends"][-1])
            self.data_dict.add("meta/episode_name", str(self.zf_in["meta/episode_name"][ep_i-1]))
    
    def process_mask(self):
        # ep_starts_ends = [0]
        # ep_starts_ends.extend(self.zf_in["meta/episode_ends"])
        # self.ids = []

        # for ep_i in tqdm.tqdm(range(1, len(ep_starts_ends)), "episode #"):
        #     # self.copy_index(ep_starts_ends[ep_i-1]) 
        #     ep_len = 0
        #     # assumes all topics have the same len
        #     for i in tqdm.tqdm(range(ep_starts_ends[ep_i-1], ep_starts_ends[ep_i]), f"timestamp in ep {ep_i}", leave=False):
        #         if self.mask[i] == 1:
        #             self.ids.append(i)
        #             ep_len += 1


        self.ids = []
        for i in range(len(self.mask)):
            if self.mask[i] == 1:
                self.ids.append(i)
        self.ids = np.array(self.ids)
    
    # def get_plateau_mask(self):
    #     ep_starts_ends = [0]
    #     ep_starts_ends.extend(self.zf_in["meta/episode_ends"])

    #     for ep_i in tqdm.tqdm(range(1, len(ep_starts_ends)), "episode #"):
    #         self.copy_index(ep_starts_ends[ep_i-1]) 
    #         ep_len = 1
    #         # assumes all topics have the same len
    #         for i in tqdm.tqdm(range(ep_starts_ends[ep_i-1], ep_starts_ends[ep_i]), f"timestamp in ep {ep_i}", leave=False):
    #             if np.all(self.zf_in["data/robot_act"][i] == self.zf_in["data/robot_act"][i-1]):
    #                 self.mask[]

if __name__ == "__main__":
    # print("postprocessing ......")
    np.set_printoptions(threshold=100)

    parser = argparse.ArgumentParser(description = "A post-processor script to take the processed zarr file and augment it for training a Diffusion Policy model")
    parser.add_argument("-d", "--data", metavar="PATH_TO_PROCESSED_ZARR",help = "path to the processed zarr file",
                        required=True)
    parser.add_argument("-p", "--processed", metavar="PATH_TO_PROCESSED_FILE", help = "path to save the post processed zarr file at",
                         required=True)
    parser.add_argument("-a", "--aux", help = "should the model learn auxilary tasks", choices=["true", "false"],
                         required=False, default = "true")
    argument = parser.parse_args()

    zarr_in_path = argument.data
    processed_path = argument.processed
    use_aux = argument.aux

    # # optional args21728129481 - 21693698127
    zarr_proc = ZarrPostProcessor(zarr_in_path, use_aux == 'true')
    # zarr_proc.copy_index(1)
    zarr_proc.remove_plateaus()
    zarr_proc.process_mask()

    
    print(zarr_proc.data_dict)
    
    print("saving processed dataset dict to", processed_path)
    # print(zarr_proc.mask)
    # zarr_proc.zf_in = None

    zf_out = zarr.ZipStore(processed_path, mode="w")
    h5_out = h5py.File("data/victor/tmp//ds_processed2.h5", mode="w")
    ep_ends = [0]
    ep_ends.extend(zarr_proc.data_dict["meta/episode_ends"])
    old_ep_ends = [0]
    old_ep_ends.extend(zarr_proc.zf_in["meta/episode_ends"])
    for ep_i in range(1, len(ep_ends)):
        print(f"SAVING EP{ep_i}")
        for t in zarr_proc.topics:
            ep_ids = zarr_proc.ids[np.where((zarr_proc.ids >= old_ep_ends[ep_i-1]) & (zarr_proc.ids < old_ep_ends[ep_i]))] - old_ep_ends[ep_i-1]
            v = np.array(zarr_proc.zf_in[t][old_ep_ends[ep_i-1]:old_ep_ends[ep_i]])[ep_ids]
            if ep_i == 1:
                zarr.array(data=v, path=t, store=zf_out, chunks=zarr_proc.zf_in[t].chunks)

                # save h5
                max_shape = [None]
                if np.array(v).ndim > 1:
                    max_shape.extend(np.array(v).shape[1:])

                max_shape = tuple(max_shape)

                h5_out.create_dataset(t, data=v, maxshape=max_shape)
            else:
                zarr.open_array(zf_out, path=t).append(v, axis=0)
                old_shape = h5_out[t].shape
                # print("OLD SHAPE", old_shape)
                h5_out[t].resize(old_shape[0] + np.array(v).shape[0], axis=0)
                # print("NEW SHAPE", hf[k].shape)
                h5_out[t][old_shape[0]:] = v    # store_zarr_dict_diff_data(processed_path, zarr_proc.data_dict)

    zarr.array(data=zarr_proc.data_dict["meta/episode_ends"], path="meta/episode_ends", store=zf_out)
    zarr.array(data=zarr_proc.data_dict["meta/episode_name"], path="meta/episode_name", store=zf_out)
    h5_out.create_dataset("meta/episode_ends", data=zarr_proc.data_dict["meta/episode_ends"])
    h5_out.create_dataset("meta/episode_name", data=zarr_proc.data_dict["meta/episode_name"])
    h5_out.close()
