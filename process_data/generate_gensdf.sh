#!/bin/bash


python 2.1_generate_gensdf_dataset.py           \
            --clear_temp_file False             \
            --n_process 22                      \
            --n_sample_point_each 1000000        \
            --uniform_sample_ratio 0.5              \
            --n_point_cloud 500000                  \
            --near_surface_sammple_method random    \
            --on_surface_sample_method poisson_disk \

# 3e5 point cloud
# 1e6 * 0.5 = 5e5 point near surface sample
# 1e6 * 0.5 = 5e5 point uniform sample
# number of point cloud may not be exactly 5e5 due to the impl. of `point_could_utils`