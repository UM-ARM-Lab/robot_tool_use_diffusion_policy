import collections
import numpy as np
from typing import List, Tuple, Optional, Dict

# TODO include timestamps to calculate latency
# Class that keeps track of the last T_o observations
class ObsAccumulator:
    def __init__(self, To):
        self.obs_dq = collections.deque(maxlen=To)  # stores the last To observations
        self.To = To

    def put(self, data: Dict[str, np.ndarray]):
        # TODO GET RID OF THIS
        # self.obs_dq = collections.deque(maxlen=self.To)  # stores the last To observations
        self.obs_dq.append(data)
        # fill the dq with the initial data point at the start
        while len(self.obs_dq) < self.To:   
            self.obs_dq.append(data)

    # turns the deque into the dictionary format that the policy expects
    # "key0": Tensor of shape (B,To,*)
    # "key1": Tensor of shape e.g. (B,To,H,W,3)
    # assumes B = 1 for robot obs (unsure how any other value could make sense)
    def get(self) -> Dict[str, np.ndarray]:
        obs_dict = {}
        for dq_dict in self.obs_dq:
            for k, v in dq_dict.items():
                if k not in obs_dict.keys():
                    obs_dict[k] = []
                obs_dict[k].append(v)
        
        for k, vlist in obs_dict.items():
            obs_dict[k] = np.expand_dims(np.stack(obs_dict[k]), axis=0)

        return obs_dict

    def __repr__(self):
        return self.obs_dq.__repr__()

if __name__ == "__main__":
    print("deck")
    oa = ObsAccumulator(2)
    oa.put({"a" : np.array([1,2,3]), "b" : np.array([[5,6,7,12], [1,1,1,1]])})
    oa.put({"a" : np.array([1,5,1]), "b" : np.array([[5,6,7,13], [2,2,2,2]])})
    oa.put({"a" : np.array([9,2,3]), "b" : np.array([[5,6,7,14], [3,3,3,3]])})
    print(oa)
    for k, v in oa.get().items():
        print("key", k, ":\n", v, "\tshape:", v.shape)