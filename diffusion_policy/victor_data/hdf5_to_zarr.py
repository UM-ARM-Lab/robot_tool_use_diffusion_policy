import zarr
import numpy as np
# import zarr.storage
import h5py
from .data_utils import get_h5_topics

if __name__ == "__main__":
    print("bi-zarr-e")
    print(zarr.__version__)
    np.set_printoptions(suppress=True, precision=5)

    h5d_path = "datasets/test_ds/ds_processed.h5"
    h5f = h5py.File(h5d_path, 'r')
    # print(get_h5_topics(h5f, ""))

    print("Converting", h5d_path, "...")

    ga = zarr.open("datasets/d1.zarr", mode="w")
    for k in get_h5_topics(h5f, ""):
        data = np.array(h5f[k])
        g = ga.create(k, mode="w", shape=data.shape, dtype=data.dtype)
        g[:] = data
        # print(g)
    # ga["timestamp"].append([777])
    print(ga.tree())
    # print()
    # print(np.array(ga["right_arm/action/data"]))
    print(np.array(ga["data/timestamp"]))
