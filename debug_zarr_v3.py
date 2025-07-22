#!/usr/bin/env python3

import zarr
import os

# Try different approaches to open the v3 zarr file
zarr_path = 'data/victor/output_data_1.zarr'

print(f"Trying to open {zarr_path}")
print(f"Path exists: {os.path.exists(zarr_path)}")

# Approach 1: Try with different zarr storage options
try:
    print("\n--- Approach 1: Standard zarr.open ---")
    store = zarr.open(zarr_path, mode='r')
    print(f"Success! Store: {store}")
    print(f"Keys: {list(store.keys())}")
except Exception as e:
    print(f"Failed: {e}")

# Approach 2: Try opening as DirectoryStore
try:
    print("\n--- Approach 2: DirectoryStore ---")
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.open(store, mode='r')
    print(f"Success! Root: {root}")
    print(f"Keys: {list(root.keys())}")
except Exception as e:
    print(f"Failed: {e}")

# Approach 3: Try with zarr v3 compatibility mode
try:
    print("\n--- Approach 3: Force v2 mode ---")
    store = zarr.open(zarr_path, mode='r', zarr_version=2)
    print(f"Success! Store: {store}")
    print(f"Keys: {list(store.keys())}")
except Exception as e:
    print(f"Failed: {e}")

# Approach 4: Check if we can use the zip version instead
zip_path = 'data/victor/output_data_1.zarr.zip'
try:
    print(f"\n--- Approach 4: Try zip version {zip_path} ---")
    print(f"Zip path exists: {os.path.exists(zip_path)}")
    store = zarr.open(zip_path, mode='r')
    print(f"Success! Store: {store}")
    print(f"Keys: {list(store.keys())}")
except Exception as e:
    print(f"Failed: {e}")
