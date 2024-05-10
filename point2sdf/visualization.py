import pyvista as pv
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

def visualization_as_pointcloud(decoder, latent, out_file, device):
    decoder.eval()
    batch_size = latent.size(0)
    points = torch.rand(batch_size, 100000, 3) - 0.5
    points = points.to(device)
    with torch.no_grad():
        occs = decoder(latent, points)

    for i in range(batch_size):
        point = points[i, ...]
        occ = occs[i, ...]
        # print('occ', type(occ), occ.shape)
        # print('point', type(point), point.shape)
        if not (occ > 0).sum():
            print('skip sample id = ', i)
            continue

            # Use numpy

        point_cpu = point[occ > 0, :].cpu().numpy()
        # Create plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(point_cpu[:, 2], point_cpu[:, 0], point_cpu[:, 1])
        ax.set_xlabel('Z')
        ax.set_ylabel('X')
        ax.set_zlabel('Y')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.view_init(elev=30, azim=45)
        if out_file is not None:
            plt.savefig(Path(out_file) / f'{i}.png')
        else:
            plt.show()
        plt.close(fig)



