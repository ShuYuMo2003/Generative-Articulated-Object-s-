shape_path = '/mnt/d/Research/data/partnet-mobility-v0/dataset/3398'
import json
import numpy as np
import pyvista as pv

shape_mobility = json.load(open(shape_path + '/mobility_v2.json'))
result = json.load(open(shape_path + '/result.json'))

# 假设你的点存储在一个名为points的numpy数组中
points = np.random.rand(100, 3)  # 生成一些随机点

# 创建一个PointCloud对象
cloud = pv.PolyData(points)

# 创建一个plotter对象，并添加你的点云
plotter = pv.Plotter()
plotter.add_mesh(cloud, color="red", point_size=5.0)

# 添加坐标轴
plotter.add_axes()

# 显示图形
plotter.show()