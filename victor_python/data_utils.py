from collections import deque
import glob
from typing import Dict, Callable
import os
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
import torch.nn as nn
import multiprocessing as mp
import ctypes
import zarr 
import tqdm


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    elif isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [to_numpy(v) for v in x]
    else:
        return x


def readOBJ(file):
    V, Vt, F, Ft = [], [], [], []
    with open(file, 'r') as f:
        T = f.readlines()
    for t in T:
        # 3D vertex
        if t.startswith('v '):
            v = [float(n) for n in t.replace('v ', '').split(' ')]
            V += [v]
        # UV vertex
        elif t.startswith('vt '):
            v = [float(n) for n in t.replace('vt ', '').split(' ')]
            Vt += [v]
        # Face
        elif t.startswith('f '):
            idx = [n.split('/') for n in t.replace('f ', '').split(' ')]
            f = [int(n[0]) - 1 for n in idx]
            F += [f]
    V = np.array(V, np.float32)
    Vt = np.array(Vt, np.float32)
    if Ft:
        assert len(F) == len(
            Ft
        ), 'Inconsistent .obj file, mesh and UV map do not have the same number of faces'
    else:
        Vt, Ft = None, None
    F = np.array(F, np.int32)
    return V, F, Vt, Ft


def read_h5_dict(path, data_names=None, exclude_names=[]):
    hf = h5py.File(path, 'r')
    data = SmartDict()
    hf_names = hf.keys()
    data_names = hf_names if data_names is None else data_names
    for name in data_names:
        if name in exclude_names or name not in hf_names:
            continue
        d = np.array(hf.get(name))
        data[name] = d
    hf.close()
    return data

def get_h5_topics(h5f, parent_str, depth = 0, max_depth = 999):
    list = []
    return __get_h5_topics_rec(h5f, parent_str, list, depth, max_depth)

def get_h5_leaf_topics(h5f, parent_str, topic_list, depth = 0, max_depth = 999):
    list = []
    return __get_h5_leaf_topics_rec(h5f, parent_str, list, depth, max_depth)


def __get_h5_topics_rec(h5f, parent_str, topic_list, depth = 0, max_depth = 999):
    try:
        ks = None
        if len(parent_str) > 0:
            ks = h5f.get(parent_str).keys()
        else:
            ks = h5f.keys()
        for k in ks:
            if depth < max_depth:
                __get_h5_topics_rec(h5f, parent_str + "/" + k, topic_list, depth + 1, max_depth)
            else:
                topic_list.append(parent_str)
                break
        return topic_list
    except:
        topic_list.append(parent_str)
    
def __get_h5_leaf_topics_rec(h5f, parent_str, topic_list, depth = 0, max_depth = 999):
    try:
        ks = None
        if len(parent_str) > 0:
            ks = h5f.get(parent_str).keys()
        else:
            ks = h5f.keys()
        for k in ks:
            if depth < max_depth:
                __get_h5_leaf_topics_rec(h5f, k, topic_list, depth + 1, max_depth)
            else:
                break
        return topic_list
    except:
        topic_list.append(parent_str)

def read_h5_dict_recursive(h5f, parent_str, dict):
    try:
        ks = None
        if len(parent_str) > 0:
            ks = h5f.get(parent_str).keys()
        else:
            ks = h5f.keys()
        for k in ks:
            read_h5_dict_recursive(h5f, parent_str + "/" + k, dict)
        return dict
    except:
        dict[parent_str] = np.array(h5f.get(parent_str))

def read_zarr_dict_recursive(z_obj, dict):
    if isinstance(z_obj, zarr.hierarchy.Group):
        for k in z_obj.keys():
            read_zarr_dict_recursive(z_obj[k], dict)
        return # do not print the groups

    dict[z_obj.name] = np.array(z_obj)


# def get_zarr_topics(zf, arr):


# def read_h5_dict(path, data_names=None, exclude_names=[]):
#     hf = h5py.File(path, 'r')
#     data = SmartDict()
#     for hk in hf.keys():
#         name_stack = [hk]
#         while len(name_stack) > 0:
#             try 
#     name_stack = [k for k in hf.keys()]


def store_h5_dict(path, data_dict, compression="lzf", **ds_kwargs):
    hf = h5py.File(path, 'w')
    for k, v in data_dict.items():
        hf.create_dataset(k, data=v, compression=compression, **ds_kwargs)
    hf.close()

def read_h5_dict_dir(path_pattern, data_names=None, exclude_names=[]):
    paths = glob.glob(path_pattern)
    datas = []
    for path in paths:
        datas.append(read_h5_dict(path, data_names, exclude_names))
    return datas

def store_zarr_dict(path, data_dict, **ds_kwargs):
    with zarr.ZipStore(path=path, mode="w") as zf:
        for k, v in data_dict.items():
            zarr.array(data=v, path=k, store=zf)

# a method for storing zarr dicts specifically for the diffusion policy training
# (using proper compression and chunking)
def store_zarr_dict_diff_data(path, data_dict, **ds_kwargs):
    with zarr.ZipStore(path=path, mode="w") as zf:
        for k, v in tqdm.tqdm(data_dict.items(), "Storing zarr dict into " + path):
            v = np.array(v)
            print(k)
            if v.ndim == 1:
                zarr.array(data=v, path=k, store=zf, chunks=(1000))
            else:
                if k == 'data/image':
                    chunk_shape = [1]
                    chunk_shape.extend(np.array(v).shape[1:])
                    chunk_shape = tuple(chunk_shape)
                    # print(chunk_shape)
                    zarr.array(data=v, path=k, store=zf, chunks=chunk_shape, dtype="u1")#, compressor=Jpeg2k(level=50))
                else:
                    chunk_shape = [1000]
                    chunk_shape.extend(np.array(v).shape[1:])
                    chunk_shape = tuple(chunk_shape)
                    zarr.array(data=v, path=k, store=zf, chunks=chunk_shape)

def find_best_checkpoint(model_path, epoch=None):
    # search.ckpt or .pth
    if '.ckpt' not in model_path and '.pth' not in model_path and ".pt" not in model_path:
        ckpt_pattern = f'{model_path}/*.ckpt'
        all_ckpts = glob.glob(ckpt_pattern)
        if len(all_ckpts) == 0:
            ckpt_pattern = f'{model_path}/*.pt'
            all_ckpts = glob.glob(ckpt_pattern)
    else:
        ckpt_pattern = model_path
        all_ckpts = glob.glob(ckpt_pattern)
    best_ckpt = sorted(all_ckpts, key=os.path.getmtime)[-1]
    print('Found checkpoint', best_ckpt)
    return best_ckpt


def aggre_dicts(dicts, stack=False):
    out = {}
    for k in dicts[0].keys():
        out[k] = [d[k] for d in dicts]
        if stack:
            out[k] = np.stack(out[k], axis=0)
    return out


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


class MetricDict(dict):
    def __init__(self, is_detach=True, store_accumulated=False):
        """
        Aggregate numpy arrays or pytorch tensors
        :param is_detach: Whether to save numpy arrays in stead of torch tensors
        """
        super(MetricDict).__init__()
        self.is_detach = is_detach
        self.count = {}
        self.store_accumulated = store_accumulated

    def __getitem__(self, item):
        return self.get(item, 0)

    def add(self, key, value):
        if self.is_detach and torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        if key not in self.keys():
            if self.store_accumulated:
                self[key] = value
                self.count[key] = 1
            else:
                self[key] = [value]
                self.count[key] = 1
        else:
            if self.store_accumulated:
                self[key] += value
                self.count[key] += 1
            else:
                self[key].append(value)
                self.count[key] += 1


    def get_mean(self, prefix=""):
        avg_dict = {}
        for k, v in self.items():
            if self.store_accumulated:
                avg_dict[prefix + k] = v / self.count[k]
            else:
                avg_dict[prefix + k] = np.mean(v)
        return avg_dict

    def __repr__(self, level=0):
        # print the mean of the metrics
        avg_dict = self.get_mean()
        ret = ""
        for k, v in avg_dict.items():
            ret += "  " * level + f"{k}: {v}\n"
        return ret


class SmartDict(Dict):
    """
    A slicable dictionary for numpy and tensor object.
    d = {"a": torch.ones(8, 3), "b": torch.ones(8, 4)}
    custom_dict = SlicedDict(d)
    sliced_result = custom_dict[:3]
    sliced_result = custom_dict[:3, 1]
    sliced_result = custom_dict[0, 0]
    sliced_result = custom_dict[:3, 1:2]
    """

    def __init__(self, *args, backend="torch", dtype="float", **kwargs):
        """Transform a nested Dict to SmartDict
        
        Args:
            *args: Arguments to pass to Dict constructor
            backend (str): The backend to use for tensor operations ("numpy" or "torch" or None)
            dtype (str): Data type to use ("float" or "half")
            **kwargs: Keyword arguments to pass to Dict constructor
        """
        super().__init__(*args, **kwargs)
        self.backend = backend
        if backend == "torch":
            self.dtype = torch.float if dtype == "float" else torch.half
        elif backend == "numpy":
            self.dtype = np.float32 if dtype == "float" else np.float16
        else:
            raise ValueError(f"Invalid backend: {backend}")
        # Transform nested dicts to SmartDict with same backend
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = SmartDict(v, backend=backend, dtype=dtype)

    def __repr__(self, level=0):
        """Print the nested structure of the SmartDict"""
        ret = ""
        for key, value in self.items():
            if isinstance(value, dict):
                ret += "  " * level + f"{key}:\n"
                ret += value.__repr__(level + 1)
            elif torch.is_tensor(value) or isinstance(value, np.ndarray):
                ret += "  " * level + f"{key}: {value.shape}  {value.dtype}\n"
            elif isinstance(value, list):
                ret += "  " * level + f"{key}: list {len(value)}\n"
            elif isinstance(value, int) or isinstance(value, float):
                ret += "  " * level + f"{key}: scalar\n"
        return ret

    def __str__(self):
        return self.__repr__()

    def apply(self, func):
        return dict_apply(self, func)

    @property
    def device(self):
        for v in self.values():
            if torch.is_tensor(v):
                return v.device
        return None

    def _to_shared_memory(self):
        # Convert arrays in self.dataset to shared memory arrays, preserving dtype
        np_to_ctypes = {
            np.dtype('float16'): ctypes.c_float,
            np.dtype('float32'): ctypes.c_float,
            np.dtype('float64'): ctypes.c_double,
            np.dtype('int32'): ctypes.c_int32,
            np.dtype('int64'): ctypes.c_int64,
            np.dtype('uint8'): ctypes.c_uint8,
            np.dtype('int8'): ctypes.c_int8,
            np.dtype('uint16'): ctypes.c_uint16,
            np.dtype('int16'): ctypes.c_int16,
            # Add more types as needed
        }
        for key in self:
            arr = self[key]
            if torch.is_tensor(arr):
                arr = arr.share_memory_()
            elif isinstance(arr, np.ndarray):
                arr_dtype = arr.dtype
                if arr_dtype not in np_to_ctypes:
                    raise TypeError(f"Unsupported dtype {arr_dtype} for shared array.")
                c_type = np_to_ctypes[arr_dtype]
                shared_array_base = mp.Array(c_type, int(np.prod(arr.shape)))
                shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
                shared_array = shared_array.reshape(arr.shape)
                np.copyto(shared_array, arr)
                self[key] = shared_array

    def __len__(self):
        for v in self.values():
            if hasattr(v, "__len__"):
                return len(v)
        return 0

    def __setitem__(self, key, value):
        if isinstance(key, int):
            raise TypeError("Integer keys are not allowed in SmartDict")
        if isinstance(key, str):
            super().__setitem__(key, value)
        elif isinstance(key, slice) or isinstance(key, tuple):
            assert len(value) == len(
                self
            ), "The length of the value should be the same as the length of the dict"
            for k, v in self.items():
                v[key] = value[k]

    def __getitem__(self, key):
        if (
            isinstance(key, slice)
            or torch.is_tensor(key)
            or isinstance(key, np.ndarray)
            or isinstance(key, tuple)
            or key is None
            or isinstance(key, int)
        ):
            sliced_dict = SmartDict()
            # check if the values are of the same length
            if key is not None:
                lens = [len(v) for v in self.values()]
                assert len(set(lens)) == 1, "Try to slice a SmartDict with varied length of values."

            for k, v in self.items():
                sliced_dict[k] = v[key]
            return sliced_dict
        else:
            return super().__getitem__(key)

    def add(self, key, value):
        if torch.is_tensor(value) and self.backend == "numpy":
            value = value.detach().cpu().numpy().astype(self.dtype)
        elif isinstance(value, np.ndarray) and self.backend == "torch":
            value = torch.from_numpy(value).type(self.dtype)
        if key not in self.keys():
            self[key] = [value]
        else:
            self[key].append(value)

    # like add, but only adds once
    def add_static(self, key, value):
        if torch.is_tensor(value) and self.backend == "numpy":
            value = value.detach().cpu().numpy().astype(self.dtype)
        elif isinstance(value, np.ndarray) and self.backend == "torch":
            value = torch.from_numpy(value).type(self.dtype)
        if key not in self.keys():
            self[key] = [value]
        # else: # for static values, we only want to record them once
        #     self[key].append(value)

    def to_numpy(self, dtype=np.float32):
        new_dict = SmartDict()
        for k, v in self.items():
            if torch.is_tensor(v):
                new_dict[k] = v.detach().float().cpu().numpy().astype(dtype)
            elif isinstance(v, SmartDict):
                new_dict[k] = v.to_numpy(dtype)
            else:
                new_dict[k] = v.astype(dtype)
        return new_dict

    def to_tensor(self, device="cuda", dtype=torch.float32):
        new_dict = SmartDict()
        for k, v in self.items():
            if isinstance(v, np.ndarray):
                new_dict[k] = torch.tensor(v, device=device, dtype=dtype)
            elif torch.is_tensor(v):
                new_dict[k] = v.to(device, dtype=dtype)
            elif isinstance(v, Dict) or isinstance(v, SmartDict):
                new_dict[k] = v.to_tensor(device, dtype)
            elif isinstance(v, list):
                if isinstance(v[0], SmartDict):
                    new_dict[k] = [x.to_tensor(device, dtype) for x in v]
                elif isinstance(v[0], np.ndarray) or isinstance(v[0], int):
                    new_dict[k] = [torch.tensor(x, device=device, dtype=dtype) for x in v]
            else:
                new_dict[k] = v
        return new_dict

    def to(self, device):
        return self.to_tensor(device)

    def clone(self):
        """Deep copy of the SmartDict"""
        return self.apply(lambda x: x.clone() if torch.is_tensor(x) else x.copy())

    def copy(self):
        """Shallow copy of the SmartDict"""
        return SmartDict(super().copy())
    
    def __add__(self, other):
        out = SmartDict()
        for k in self.keys():
            out[k] = self.get(k, 0) + other[k]
        return out

    def __iadd__(self, other):
        for k in other.keys():
            self[k] = self.get(k, 0) + other[k]
        return self

    def __div__(self, other: float | Dict):
        out = SmartDict()
        for k in self.keys():
            out[k] = self.get(k, 0) / other
        return out

class SmartQueue(SmartDict):
    def __init__(self, max_size, reverse_queue=False, *args, **kwargs):
        """
        A queue that can be sliced and stacked.
        reverse_queue: If True, the queue is reversed. New item is added to the front of the queue.
        """
        super().__init__(*args, **kwargs)
        self.max_size = max_size
        self.reverse_queue = reverse_queue

    def add(self, key, value):
        if torch.is_tensor(value) and self.backend == "numpy":
            value = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray) and self.backend == "torch":
            value = torch.from_numpy(value)
        if key not in self.keys():
            self[key] = [value] * self.max_size
        else:
            if self.reverse_queue:
                self[key].pop()
                self[key].insert(0, value)
            else:
                self[key].pop(0)
                self[key].append(value)

def stack_dict(dicts, axis=0):
    out = SmartDict()
    for k in dicts[0].keys():
        stackable = False
        if torch.is_tensor(dicts[0][k]):
            stack = torch.stack
            stackable = True
        elif isinstance(dicts[0][k], np.ndarray):
            stack = np.stack
            stackable = True
        elif isinstance(dicts[0][k], SmartDict):
            stack = stack_dict
            stackable = True
        if stackable:
            out[k] = stack([d[k] for d in dicts], axis)
        else:
            out[k] = [d[k] for d in dicts]
    return out


def cat_dict(dicts, axis=0):
    out = SmartDict()
    for k in dicts[0].keys():
        catable = False
        if torch.is_tensor(dicts[0][k]):
            cat = torch.cat
            catable = True
        elif isinstance(dicts[0][k], np.ndarray):
            cat = np.concatenate
            catable = True
        elif hasattr(dicts[0][k], 'items'):
            cat = cat_dict
            catable = True
        if catable:
            out[k] = cat([d[k] for d in dicts], axis)
        else:
            out[k] = [d[k] for d in dicts]
    return out


def dict_apply(
    x: Dict[str, torch.Tensor], func: Callable[[torch.Tensor], torch.Tensor]
) -> Dict[str, torch.Tensor]:
    result = SmartDict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def dict_list_to_list_dict(dict_list):
    list_dict = {}
    for k in dict_list[0].keys():
        list_dict[k] = []
    for d in dict_list:
        for k, v in d.items():
            list_dict[k].append(v)
    return list_dict


def torch_stack_list_dict(list_dict):
    for k, v in list_dict.items():
        if len(v) > 0:
            list_dict[k] = torch.stack(v, dim=0)
    return list_dict

