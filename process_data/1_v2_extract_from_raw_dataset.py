import json
import shutil
import time
import numpy as np
import pyvista as pv
from tqdm import tqdm
from rich import print
from glob import glob
from pathlib import Path


import sys
sys.path.append('..')
from utils.utils import HighPrecisionJsonEncoder


def degree2rad(degree):
    # assert -180 <= degree <= 180
    return degree * np.pi / 180

def parse_partid_to_objs(shape_path:Path):
    result_file_path = shape_path / 'result.json'
    result_file = json.loads(result_file_path.read_text())
    partid_to_objs = {}
    def parse_part(part):
        pid = part['id']
        partid_to_objs[pid] = set(part.get('objs', set()))
        for child in part.get('children', []):
            parse_part(child)
            childs_objs = partid_to_objs[child['id']]
            partid_to_objs[pid] |= childs_objs

    assert len(result_file) == 1
    parse_part(result_file[0])

    return partid_to_objs

def merge_meshs(meshs_paths: list[Path], output_path:Path):
    meshs_paths = list(set(meshs_paths))
    meshs = []
    for mesh_path in meshs_paths:
        try:
            value = pv.read(str(mesh_path))
            meshs.append(value)
        except FileNotFoundError:
            print(f"[Warning] {mesh_path} not found.")

    merged_mesh = meshs[0]
    for mesh in meshs[1:]:
        merged_mesh += mesh
    merged_mesh.save(str(output_path))
    return merged_mesh

def parse_limit(part_data):
    if part_data['parent'] == -1 or part_data['joint'] in ['junk', 'fixed']:
        return [0., 0., 0., 0.]
    limit = part_data['jointData']['limit']
    if part_data['joint'] == 'hinge':     # ======= 仅旋转关节 =========
        if limit['noLimit']:
            return [0, 0, -np.pi, np.pi]
        else:
            return [0, 0, degree2rad(limit['a']), degree2rad(limit['b'])]
    elif part_data['joint'] == 'slider':
        if not limit.get('rotates'): # ======= 仅滑动关节 =========
            return [limit['a'], limit['b'], 0, 0]
        else:
            if limit['noRotationLimit']:
                return [limit['a'], limit['b'], -np.pi, np.pi]
            else:                # ======= 旋转滑动关节 =========
                rotate_limit = limit['rotationLimit']
                rotate_limit = degree2rad(rotate_limit)
                return [limit['a'], limit['b'], -rotate_limit, rotate_limit]
    else:
        raise NotImplementedError

dfn = 0
id_to_dfn = {}
def calcuate_dfn(parts, cur_id):
    global dfn
    child = list(filter(lambda x: x['raw_parent'] == cur_id, parts))
    dfn += 1
    id_to_dfn[cur_id] = dfn
    for c in child:
        calcuate_dfn(parts, c['raw_id'])


def process(shape_path:Path, output_info_path:Path, output_mesh_path:Path, needed_categories:list[str]):

    start_time      = time.time()
    raw_meta_path   = Path(shape_path) / 'meta.json'
    raw_meta        = json.loads(raw_meta_path.read_text())
    mobility_file_path = Path(shape_path) / 'mobility_v2.json'
    mobility_file   = json.loads(mobility_file_path.read_text())

    catecory_name   = raw_meta['model_cat']
    meta            = {'catecory': raw_meta['model_cat'], 'shape_id': raw_meta['anno_id'], 'shape_path': shape_path.as_posix()}
    processed_part  = []

    if catecory_name not in needed_categories and '*' not in needed_categories:
        return f"[Skip] {catecory_name} is not in needed categories."

    print('Processing:', shape_path)

    partid_to_objs = parse_partid_to_objs(shape_path)
    # print('parse_partid_to_objs', partid_to_objs)

    for part in mobility_file:
        new_part = {}
        pid = part['id']

        new_part['raw_id'] = pid
        new_part['raw_parent'] = part['parent']

        # Get Meshs in `obj`
        partids_in_result = [obj['id'] for obj in part["parts"]]
        objs_file = set()
        for partid_in_result in partids_in_result:
            objs_file |= partid_to_objs[partid_in_result]

        meshs_paths = [shape_path / 'textured_objs' / (obj + '.obj')
                         for obj in objs_file]
        merged_mesh_name = f"{catecory_name}_{meta['shape_id']}_{pid}.ply"
        merged_mesh_save_path = output_mesh_path / merged_mesh_name
        mesh = merge_meshs(meshs_paths, merged_mesh_save_path)
        new_part['mesh'] = merged_mesh_name

        bounding_box = mesh.bounds
        new_part['bounding_box'] = [bounding_box[0::2], bounding_box[1::2]]

        if part['parent'] != -1:
            new_part['joint_data_origin'] = part['jointData']['axis']['origin']
            new_part['joint_data_direction'] = part['jointData']['axis']['direction']
            if None in new_part['joint_data_direction']:
                return f"[Error]: Bad data in {shape_path.as_posix()}"
        else:
            new_part['joint_data_origin'] = [0, 0, 0]
            new_part['joint_data_direction'] = [0, 0, 0]
        new_part['limit'] = parse_limit(part)

        processed_part.append(new_part)

    global dfn
    global id_to_dfn
    dfn = 0
    id_to_dfn = {-1: 0}
    root_cnt = 0
    for part in processed_part:
        if part['raw_parent'] == -1:
            calcuate_dfn(processed_part, part['raw_id'])
            root_cnt += 1

    if root_cnt > 1:
        return "[Error]: More than one root part."

    for part in processed_part:
        part['dfn'] = id_to_dfn[part['raw_id']]
        part['dfn_fa'] = id_to_dfn[part['raw_parent']]

    for part in processed_part:
        part.pop('raw_id')
        part.pop('raw_parent')

    output_info_path = output_info_path / f"{catecory_name}_{meta['shape_id']}.json"
    output_info_path.write_text(json.dumps({
            'meta': meta,
            'part': processed_part
        } , cls=HighPrecisionJsonEncoder, indent=2))

    end_time = time.time()
    return f'[Done] {shape_path.as_posix()} time: {end_time - start_time:.2f}s'


if __name__ == '__main__':
    raw_dataset_paths   = glob('../dataset/raw/*')
    output_info_path    = Path('../dataset/1_preprocessed_info')
    output_mesh_path    = Path('../dataset/1_preprocessed_mesh')
    needed_categories   = [
            'USB',
            # 'Chair',
            # 'Door',
            # 'StorageFurniture',
            # 'Toilet',
        ]

    shutil.rmtree(output_info_path, ignore_errors=True)
    shutil.rmtree(output_mesh_path, ignore_errors=True)

    output_info_path.mkdir(exist_ok=True)
    output_mesh_path.mkdir(exist_ok=True)

    failed_shape_path = []
    success_shape_path = []

    for shape_path in tqdm(raw_dataset_paths):

        status = process(Path(shape_path), output_info_path, output_mesh_path, needed_categories)
        if 'Error' in status:
            failed_shape_path.append(shape_path)
        elif 'Done' in status:
            success_shape_path.append(shape_path)

        if 'Skip' not in status:
            print(status)

    print('Failed shape path:', failed_shape_path)
    print('# Failed shape:', len(failed_shape_path))
    print('# Success shape:', len(success_shape_path))