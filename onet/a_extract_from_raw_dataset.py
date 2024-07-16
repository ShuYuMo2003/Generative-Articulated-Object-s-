import json
import torch
import numpy as np
import pyvista as pv
from pathlib import Path
from functools import reduce
from tqdm import tqdm
from glob import glob
from rich import print

def degree2rad(degree):
    # assert -180 <= degree <= 180
    return degree * np.pi / 180

def convert_meshs(shape_path, output_path, mesh_output_path):
    result = json.load(open(shape_path / 'result.json', 'r'))
    mobility = json.load(open(shape_path / 'mobility_v2.json', 'r'))
    mobility_dict = { part['id']: part for part in mobility }
    shape_original_id = shape_path.stem
    model_cat = json.load(open(shape_path / 'meta.json', 'r'))['model_cat']

    for value in mobility_dict.values():
        if value['joint'] == 'junk':
            print('meet junk joint.')
            return str(shape_path)

    # output_path = output_path / shape_path.stem
    # mesh_output_path = output_path / 'mesh'
    mesh_output_path.mkdir(parents=True, exist_ok=True)

    # merge related parts.
    part_id_2_objs = {}
    def process_part(state_dict):
        if state_dict['id'] not in part_id_2_objs:
            part_id_2_objs[state_dict['id']] = []
        target = part_id_2_objs[state_dict['id']]
        if state_dict.get('objs'):
            target.extend(state_dict['objs'])
        for child in state_dict.get('children', []):
            process_part(child)
            target.extend(part_id_2_objs[child['id']])

    assert len(result) == 1
    process_part(result[0])

    mobility_id_2_objs = {}

    for part in mobility:
        if part['id'] not in mobility_id_2_objs:
            mobility_id_2_objs[part['id']] = []
        target = mobility_id_2_objs[part['id']]
        for part_0 in part['parts']:
            part_id = part_0['id']
            target.extend(part_id_2_objs[part_id])

    for d in mobility_id_2_objs.keys():
        mobility_id_2_objs[d] = list(set(mobility_id_2_objs[d]))

    # calc translation
    def translation_chain(idx):
        if translation.get(idx) is None:
            translation[idx] = 0 * translation_chain(mobility_dict[idx]['parent'])  \
                 - np.array(mobility_dict[idx]['jointData']['axis']['origin'])
        return translation[idx]

    translation = {} # 子坐标系 -> 地面坐标系
    for idx in mobility_id_2_objs.keys():
        if mobility_dict[idx]['parent'] == -1:
            translation[idx] = np.array([0, 0, 0])

    for idx in mobility_id_2_objs.keys():
        translation_chain(idx)


    combined_mesh = {}
    for idx in mobility_id_2_objs.keys():
        meshs = []
        for mesh_name in mobility_id_2_objs[idx]:
            try: meshs.append(pv.read(shape_path / 'textured_objs' / (mesh_name + '.obj')))
            except FileNotFoundError: print(f"File not found: {mesh_name}")
        try: combined_mesh[idx] = reduce(lambda x, y: x + y, meshs)
        except ValueError:
            print(f"Error: {meshs}")
            return "Error on combine meshs."
        combined_mesh[idx].points += translation[idx]
        combined_mesh[idx].save(str(mesh_output_path / f'{model_cat}-{shape_original_id}-{idx}.ply'))


    overall_result = {
        idx : {
            'mesh_off': f'{model_cat}-{shape_original_id}-{idx}.ply',                          # SDF
            'bounds': np.array(combined_mesh[idx].bounds, dtype=np.float32),                                            # bounds
            'tran': np.array(translation[idx], dtype=np.float32),                                                       # 向地面坐标系的转换矩阵
            'direction': np.array(mobility_dict[idx]['jointData']['axis']['direction']               # 子坐标系下的移动方向
                        if mobility_dict[idx]['parent'] != -1 else np.array([0, 0, 0]), dtype=np.float32),
            'origin': np.array(translation[idx] - (                                                  # 父坐标系下，子坐标系的原点
                translation[mobility_dict[idx]['parent']] if mobility_dict[idx]['parent'] != -1 else 0), dtype=np.float32),

        }
        for idx in combined_mesh.keys()
    }

    # 处理边界。
    for idx in overall_result.keys():
        overall_result[idx]['limit'] = np.zeros(4, dtype=np.float32) # [平移最小值, 平移最大值, 旋转最小值, 旋转最大值]
        limit = overall_result[idx]['limit']


        if mobility_dict[idx]['parent'] == -1:
            # just set limit all 0.
            continue

        raw_limit = mobility_dict[idx]['jointData']['limit']

        try:

            if mobility[idx]['joint'] == 'hinge':                       # ======= 仅旋转
                if raw_limit['noLimit']:
                    limit[-2:] = np.array([-np.pi, np.pi])
                else:
                    limit[-2:] = np.array([degree2rad(raw_limit['a']), degree2rad(raw_limit['b'])], dtype=np.float32)
            elif mobility[idx]['joint'] == 'slider':
                if not raw_limit.get('rotates'):                        # ======= 仅平移
                    limit[:2]  = np.array([raw_limit['a'], raw_limit['b']], dtype=np.float32)
                else:                                                   # ======= 旋转 + 平移
                    # 处理旋转
                    # assert raw_limit['noRotationLimit'] # 测试: slider 结点：所有旋转都是无限制的。 [upd] partnet-mobility-v0\dataset\101336 有限制。
                    if raw_limit['noRotationLimit']:
                        limit[-2:] = np.array([-np.pi, np.pi])
                    else:
                        rlim = raw_limit['rotationLimit']
                        limit[-2:] = np.array([degree2rad(-rlim), degree2rad(+rlim)])

                    # 处理平移
                    limit[:2]  = np.array([raw_limit['a'], raw_limit['b']], dtype=np.float32)
            else:
                raise ValueError(f"Unknown joint type: {mobility[idx]['joint']}")
        except Exception as e:
            print(f"error idx = : {idx}", raw_limit, shape_path, e)
            raise e



    # 重标号，按照 dfs 序，从 1 开始标号。
    assert list(range(len(overall_result))) == list(overall_result.keys())
    child = [[] for i in range(len(overall_result))]
    # print(child)
    for idx in overall_result.keys():
        assert idx == mobility_dict[idx]['id']
        if mobility_dict[idx]['parent'] != -1:
            child[mobility_dict[idx]['parent']].append(idx)

    # print(child)

    dfn = [0] * len(overall_result)

    global dfn_idx

    dfn_idx = 0

    def dfs(idx):
        global dfn_idx
        dfn_idx = dfn_idx + 1
        dfn[idx] = dfn_idx
        for child_idx in child[idx]:
            dfs(child_idx)

    for idx in range(len(overall_result)):
        if mobility_dict[idx]['parent'] == -1:
            dfs(idx)
            break

    new_parts = []
    for idx in range(len(overall_result)):
        info = overall_result[idx]
        # 0 不存在于 dfs 序列中。
        slice = {'dfn': dfn[idx], 'dfn_fa' : dfn[mobility_dict[idx]['parent']]
                    if mobility_dict[idx]['parent'] != -1 else 0}
        slice.update(info)
        new_parts.append(slice)

    new_parts.sort(key=lambda x: x['dfn'])

    # print(new_parts)

    result = {
        'meta': {
            'idx': shape_original_id,
            'model_cat': model_cat,
        },
        'part': new_parts,
    }

    np.save(output_path / f'{model_cat}-{shape_original_id}.npy', result)

'''
{
        'dfn'       : 16,
        'dfn_fa'    : 14,
        'mesh_off'  : 'Chair-179-14.ply',
        'bounds'    : array([-0.07283054,  0.01862646, -0.04461233,  0.04463167,  0.05644448, 0.10792848], dtype=float32),
        'tran'      : array([-0.53554255,  0.78743565, -0.17255151], dtype=float32),
        'direction' : array([ 3.0658874e-01, -6.4102969e-05, -9.5184207e-01], dtype=float32),
        'origin'    : array([-0.0558273 ,  0.10212389,  0.09510827], dtype=float32),
        'limit'     : array([ 0.       ,  0.       , -3.1415927,  3.1415927], dtype=float32)
    },
'''

def toy(shape_path, output_path, mesh_output_path):
    result = convert_meshs(shape_path, output_path, mesh_output_path)
    for idx in range(1, len(result['part']) + 1):
        print(idx)
        plotter = pv.Plotter()

        plotter.add_mesh(pv.read(mesh_output_path / result['part'][idx - 1]['mesh_off']), color="blue", opacity=0.3)

        plotter.add_mesh(pv.Sphere(radius=0.04, center=np.zeros(3)), color="red")

        try: plotter.add_arrows(np.zeros(3), 0.5 * np.array(result['part'][idx - 1]['direction']), line_width=0.03, color='red')
        except KeyError: print('skip')

        start_points = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        plotter.add_arrows(start_points, vectors, line_width=0.01)

        plotter.add_axes()
        plotter.add_floor()
        plotter.show()
        plotter.close()

def main(args):
    import os
    import time
    import shutil

    raw_dataset_paths = glob(os.path.join(args['raw_dataset_path'], '*'))

    output_info_path = Path(args['output_info_path'])
    output_mesh_path = Path(args['output_mesh_path'])

    shutil.rmtree(output_info_path, ignore_errors=True)
    shutil.rmtree(output_mesh_path, ignore_errors=True)

    output_info_path.mkdir(parents=True, exist_ok=True)
    output_mesh_path.mkdir(parents=True, exist_ok=True)

    filtered_paths = []

    category = args['selected_category_name']

    for path in tqdm(raw_dataset_paths, desc='-- filtering'):
        with open(os.path.join(path, 'meta.json')) as f:
            meta = json.loads(f.read())
            if (meta['model_cat'] in category or '*' in category) and '4204' in path:
                filtered_paths.append(path)

    from multiprocessing import Pool
    failed = []
    with Pool(args['n_precessor']) as _:
        results = [(path, _.apply_async(convert_meshs, (path, output_info_path, output_mesh_path)))
                    for path in map(Path, filtered_paths)]

        bar = tqdm(total=len(results), desc='-- Processing')
        while bar.n < bar.total:
            for path, res in results:
                if res.ready():
                    bar.update()
                    return_value = res.get()
                    if isinstance(return_value, str):
                        failed.append(return_value)
                    results.remove((path, res))
                    break
            time.sleep(0.05)
        bar.close()

    print('-- failed: ', len(failed))
    print('-- failed content: ', str(failed))
    print('-- process_raw_dataset done')
