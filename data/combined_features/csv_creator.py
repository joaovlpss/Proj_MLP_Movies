import numpy as np
import os

# Assuming your .npy files are in the current directory
npy_files = [f for f in os.listdir() if f.endswith('.npy')]
arrays = []

for file in npy_files:
    array = np.load(file)
    shape = array.shape
    if len(shape) != 1 or shape[0] != 210:
        print(f"Invalid shape. Expected a single vector with 210 features, got {shape[0]} features. File name = {file}")
        continue
    # Reshape each array to 2D (1 x 123)
    array_2d = array.reshape(1, -1)
    arrays.append(array_2d)

# Concatenate along the first axis (0)
combined_array = np.concatenate(arrays, axis=0)

# Check if the combined array shape is 48 x 210     
if combined_array.shape != (100, 210):
    print(f"Unexpected combined array shape: {combined_array.shape}")

# Save the combined array
np.save('combined_data.npy', combined_array)

# Print the combined array as a 48 x 210 matrix
for row in combined_array:
    print(' '.join(map(str, row)))
