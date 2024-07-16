import numpy as np

from glob import glob


if __name__ == '__main__':
    mesh_info_paths = glob('../dataset/1_preprocessed_info/*')
    mesh_info = {
        np.load(path, allow_pickle=True)
        for path in mesh_info_paths
    }
