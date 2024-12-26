import os
import numpy as np
import torch

def get_ar_params(num_classes, file_path):
    """
    Load AR parameter lists from 'file_path' if it exists,
    otherwise generate them randomly and save.

    Generate 3*3*3 kernels for each class.
    """
    #TODO add seed param for reproducibility

    if os.path.exists(file_path):
        # Load from npz
        data = np.load(file_path, allow_pickle=True)
        b_list = data["b_list"]  # This should be a numpy object array
        print(f"Loaded AR parameters from {file_path}")
    else:
        # Generate random AR params
        b_list = []
        for _ in range(num_classes):
            b = torch.randn((3, 3, 3))
            for c in range(3):
                b[c][2][2] = 0
                b[c] /= torch.sum(b[c])
            b_list.append(b.numpy())

        # Save to .npz
        np.savez(file_path, b_list=b_list)
        print(f"Generated new AR parameters and saved to {file_path}")

    return b_list
