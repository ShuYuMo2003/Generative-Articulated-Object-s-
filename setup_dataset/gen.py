import json
import torch
import numpy as np
import pyvista as pv
from pathlib import Path
from functools import reduce
from tqdm import tqdm
from glob import glob
from pytorch3d.io import load_ply, save_obj

def degree2rad(degree):
    # assert -180 <= degree <= 180
    return degree * np.pi / 180

def convert_meshs(shape_path, output_path):
    result = json.load(open(shape_path / 'result.json', 'r'))
    mobility = json.load(open(shape_path / 'mobility_v2.json', 'r'))
    mobility_dict = {part['id']: part for part in mobility}

    for value in mobility_dict.values():
        if value['joint'] == 'junk':
            print('meet junk joint.')
            return str(shape_path)

    output_path = output_path / shape_path.stem
    mesh_output_path = output_path / 'mesh'
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
        combined_mesh[idx] = reduce(lambda x, y: x + y, meshs)
        combined_mesh[idx].points += translation[idx]
        combined_mesh[idx].save(str(mesh_output_path / f'mesh-{idx}.ply'))
        # https://stackoverflow.com/questions/77829919/pytorch3d-file-io-throws-error-attributeerror-evt
        _verts, _faces = load_ply(str(mesh_output_path / f'mesh-{idx}.ply'))
        save_obj(str(mesh_output_path / f'mesh-{idx}.obj'), _verts, _faces)

    overall_result = {
        idx : {
            'node': {
                'mesh_obj': str(mesh_output_path / f'mesh-{idx}.obj'),                                                 # SDF
                'bounds': combined_mesh[idx].bounds,                                            # bounds
                'tran': translation[idx],                                                       # 向地面坐标系的转换矩阵
                'direction': mobility_dict[idx]['jointData']['axis']['direction']               # 子坐标系下的移动方向
                            if mobility_dict[idx]['parent'] != -1 else np.array([0, 0, 0])
            },
            'from_parent': {
                'origin': translation[idx] - (                                                  # 父坐标系下，子坐标系的原点
                    translation[mobility_dict[idx]['parent']] if mobility_dict[idx]['parent'] != -1 else 0
                ),
            }
        }
        for idx in combined_mesh.keys()
    }

    # 处理边界。
    for idx in overall_result.keys():
        overall_result[idx]['from_parent']['limit'] = np.zeros(4, dtype=np.float32) # [平移最小值, 平移最大值, 旋转最小值, 旋转最大值]
        limit = overall_result[idx]['from_parent']['limit']


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

    np.save(output_path / 'data.npy', overall_result)
    np.save(output_path / 'meta.npy', {
        'meta': json.load(open(shape_path / 'meta.json', 'r')),
        'mobility': mobility_dict,
        'result': result,
        'id': shape_path.stem
    })

    return overall_result

def toy(shape_path, output_path):
    overall_result = convert_meshs(shape_path, output_path)
    for idx in overall_result.keys():
        plotter = pv.Plotter()

        plotter.add_mesh(pv.read(overall_result[idx]['node']['mesh_obj']), color="blue", opacity=0.3)

        plotter.add_mesh(pv.Sphere(radius=0.04, center=np.zeros(3)), color="red")

        try: plotter.add_arrows(np.zeros(3), 0.5 * np.array(overall_result[idx]['node']['direction']), line_width=0.03, color='red')
        except KeyError: print('skip')

        start_points = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        plotter.add_arrows(start_points, vectors, line_width=0.01)

        plotter.add_axes()
        plotter.add_floor()
        plotter.show()
        plotter.close()

if __name__ == '__main__':
    toy(Path('2780'), Path('output'))
    exit(0)
    import os
    import time
    import shutil

    raw_paths = glob("../dataset/partnet-mobility")
    output_path = Path('output')
    shutil.rmtree(output_path, ignore_errors=True)
    filtered_paths = []
    category = {'USB'}

    for path in tqdm(raw_paths, desc='filtering'):
        with open(os.path.join(path, 'meta.json')) as f:
            meta = json.loads(f.read())
            if meta['model_cat'] in category or '*' in category:
                filtered_paths.append(path)

    black_instance = ['10108', '102994']

    from multiprocessing import Pool
    failed = []
    with Pool(10) as _:
        results = [_.apply_async(convert_meshs, (path, output_path, ))
                    for path in map(Path, filtered_paths)
                        if path.stem not in black_instance]
        # for res in tqdm(results, desc='Processing'):
        #     res.get()
        bar = tqdm(total=len(results), desc='Processing')
        while bar.n < bar.total:
            for res in results:
                if res.ready():
                    bar.update()
                    return_value = res.get()
                    if isinstance(return_value, str):
                        failed.append(return_value)
                    results.remove(res)
                    break
            time.sleep(0.05)

    print(len(failed))
    print(failed)
