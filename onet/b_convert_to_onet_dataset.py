from onet.generate_train_data import *
from glob import glob
import os
import shutil
from pathlib import Path

def main(config):
    MESH_FUSION_PATH = Path(config['mesh_fusion_path'])

    selected_cate = config['selected_category_name']

    output_mesh_path = Path(config['output_mesh_path'])
    output_mesh_path.mkdir(exist_ok=True, parents=True)

    output_dataset_path = Path(config['output_dataset_path'])
    shutil.rmtree(output_dataset_path, ignore_errors=True)
    process_obj(output_mesh_path, output_dataset_path, MESH_FUSION_PATH)
    print('-- generate_onet_dataset done')
