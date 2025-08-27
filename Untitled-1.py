import numpy as np

# Path to your file
file_path = ".dataset_src/features_loop_7/closeness.npy"

# Load the array
arr = np.load(file_path, allow_pickle=True)

# Check its shape and contents
print(arr.shape)
print(arr[:5])  # first 5 elements
