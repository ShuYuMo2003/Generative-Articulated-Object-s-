#!/bin/bash


python 2_generate_onet_dataset.py       \
            --wt_method pcu             \
            --sdf_method libmesh        \
            --sdf_type occ              \
            --clear_temp_file False     \
            --n_process 18              \
            --n_point_each 100000       \
            --uniform_sample_ratio 0.6  \

