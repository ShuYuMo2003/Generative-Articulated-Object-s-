import copy
import numpy as np
import pyvista as pv
from rich import print

def calc_linear_value(L, R, ratio):
    return 1.0 * L + (R - L) * 1.0 * ratio

def produce_rotate_matrix(direction, angle):
    if not isinstance(direction, np.ndarray):
        direction = np.array(direction)
    direction = direction / np.linalg.norm(direction)
    K = np.array([
        [0, -direction[2], direction[1]],
        [direction[2], 0, -direction[0]],
        [-direction[1], direction[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    M = np.eye(4)
    M[0:3, 0:3] = R
    return M

def produce_translate_matrix(direction, distance):
    if not isinstance(direction, np.ndarray):
        direction = np.array(direction)
    M = np.eye(4)
    M[0:3, 3] = direction * distance
    return M

def produce_rotate_around_line_matrix(start, direction, angle):
    if not isinstance(start, np.ndarray):
        start = np.array(start)
    if not isinstance(direction, np.ndarray):
        direction = np.array(direction)
    T = produce_translate_matrix(-start, 1)
    R = produce_rotate_matrix(direction, angle)
    T_inv = produce_translate_matrix(start, 1)
    return T_inv @ R @ T

def generate_obj_pics(_parts_data, percentage):
    print('generate_obj_pics called with percentage = ', percentage)
    parts_data = copy.deepcopy(_parts_data)
    # Sort by dfn
    parts_data.sort(key=lambda x: x['dfn'])

    dfn_to_part = {
        part['dfn']: part
        for part in parts_data
    }

    # print(dfn_to_part)

    # Calculate the depth of each part
    for dfn, part in dfn_to_part.items():
        parent_dfn = dfn_to_part[dfn]['dfn_fa']
        if parent_dfn == 0:
            dfn_to_part[dfn]['depth'] = 0
        else:
            dfn_to_part[dfn]['depth'] = dfn_to_part[parent_dfn]['depth'] + 1

    # Sort by depth
    dfn_to_part = {
        k : v
        for k, v in sorted(dfn_to_part.items(), key=lambda x: x[1]['depth'])
    }

    for dfn, part in reversed(dfn_to_part.items()):
        if(part['dfn_fa'] == 0):
            continue
        if not dfn_to_part[part['dfn_fa']].get('subtree_child'):
            dfn_to_part[part['dfn_fa']]['subtree_child'] = []

        dfn_to_part[part['dfn_fa']]['subtree_child'].append(part['dfn'])
        dfn_to_part[part['dfn_fa']]['subtree_child'].extend(part.get('subtree_child', []))

    for dfn, part in dfn_to_part.items():
        if part['dfn_fa'] == 0:
            part['transform'] = np.eye(4)
            continue

        # print(dfn, dfn_to_part)
        child = part.get('subtree_child', [])

        slide_distance = calc_linear_value(*part['limit'][:2], percentage)
        sM = produce_translate_matrix(part['joint_data_direction'], slide_distance)

        angle = calc_linear_value(*part['limit'][2:], percentage)
        rM = produce_rotate_around_line_matrix(part['joint_data_origin'], part['joint_data_direction'], angle)

        M = rM @ sM

        to_be_apply = [part] + [dfn_to_part[c] for c in child]
        for p in to_be_apply:
            p['transform'] = M @ p.get('transform', np.eye(4))
            # print('#')
            # print(M.shape, np.array(p['joint_data_direction'] + [0]).T.shape)
            # print(np.array(p['joint_data_direction'] + [0]).T)
            # print(p)
            p['joint_data_direction'] = M @ np.array(p['joint_data_direction'] + [0]).T
            p['joint_data_origin'] = M @ np.array(p['joint_data_origin'] + [1]).T

            p['joint_data_direction'] = list(p['joint_data_direction'][:3])
            p['joint_data_origin'] = list(p['joint_data_origin'][:3])

    # print(dfn_to_part)

    meshs = []
    for dfn, part in dfn_to_part.items():
        mesh = part['mesh']
        min_bound, max_bound = mesh.bounds
        tg_min_bound, tg_max_bound = part['bounding_box']

        min_bound, max_bound = np.array(min_bound), np.array(max_bound)
        tg_min_bound, tg_max_bound = np.array(tg_min_bound), np.array(tg_max_bound)

        max_bound[max_bound - min_bound < 1e-5] += 0.01
        tg_max_bound[tg_max_bound - tg_min_bound < 1e-5] += 0.01

        mesh.vertices = tg_min_bound + (tg_max_bound - tg_min_bound) * (
            (mesh.vertices - min_bound) / (max_bound - min_bound)
        )

        mesh.vertices = np.concatenate((
            mesh.vertices,
            np.ones((mesh.vertices.shape[0], 1))
        ), axis=1)

        vertices_on_ground = part['transform'] @ mesh.vertices.T
        vertices_on_ground = vertices_on_ground[0:3, :].T
        mesh.vertices = vertices_on_ground
        meshs.append(mesh)

    # print(dfn_to_part)
    plotter = pv.Plotter(off_screen=True)
    for (idx, mesh) in enumerate(meshs):
        plotter.add_mesh(mesh, color=['white', 'red', 'green', 'blue'][idx % 4])

    plotter.add_axes()
    plotter.camera_position = [
            (-3.8362383391098698, 0.32839933941769145, 2.3169300384519738),
            (-0.04570716149497277, -0.06563260832821388, -0.06195879116203942),
            (0.06658921116808907, 0.9960418054055029, -0.05887782978811195)
        ]
    plotter.show()
    buffer = plotter.screenshot()
    plotter.close()

    for idx, mesh in enumerate(meshs):
        # save mesh to file
        mesh.export(f'mesh{idx}.obj', file_type='obj')

    return buffer

'''
def gen_obj_pic(parts, percentage):
    meshs = []
    parts.sort(key=lambda x: x['dfn'])

    dfn_to_part = {
        part['dfn'] : part
        for part in parts
    }

    print(dfn_to_part)

    for dfn, part in dfn_to_part.items():
        parent_dfn = dfn_to_part[dfn]['dfn_fa']
        dfn_to_part[dfn]['parent'] = dfn_to_part.get(parent_dfn)
        dfn_to_part[dfn]['ground_to_here_transform'] = np.eye(4)
        dfn_to_part[dfn]['additional_transform_for_child'] = np.eye(4)

    for dfn, part in dfn_to_part.items():
        if dfn == 1: continue
        current_part = dfn_to_part[dfn]
        parent_part = current_part['parent']

        parent_to_current_transform = np.eye(4)
        parent_to_current_transform[0:3, 3] = -current_part['origin']
        current_part['ground_to_here_transform'] = (
            parent_part['additional_transform_for_child'] @
            parent_to_current_transform @
            parent_part['ground_to_here_transform']
        )

        # apply slice
        slice_distance = calc_linear_value(*current_part['limit'][:2], percentage)
        additional_transform_for_child = np.eye(4)
        additional_transform_for_child[0:3, 3] = slice_distance * current_part['direction']

        # apply rotate.
        angle = calc_linear_value(*current_part['limit'][2:], percentage)
        M = produce_rotate_matrix(current_part['direction'], angle)
        additional_transform_for_child = M @ additional_transform_for_child

        current_part['additional_transform_for_child'] = additional_transform_for_child



    for dfn, part in dfn_to_part.items():
        mesh = part['mesh_off']
        min_bound, max_bound = mesh.bounds

        tg_min_bound = part['bounds'][0::2]
        tg_max_bound = part['bounds'][1::2]

        mesh.vertices = tg_min_bound + (tg_max_bound - tg_min_bound) * (
                (mesh.vertices - min_bound) / (max_bound - min_bound)
            )


        vertices = np.concatenate((
                mesh.vertices,
                np.ones((mesh.vertices.shape[0], 1))
            ), axis=1)

        vertices_on_ground = vertices @ part['ground_to_here_transform'].T
        vertices_on_ground = vertices_on_ground[:, 0:3]
        mesh.vertices = vertices_on_ground
        meshs.append(mesh)

    print(dfn_to_part)

    plotter = pv.Plotter()
    for (idx, mesh) in enumerate(meshs):
        plotter.add_mesh(mesh, color=['white', 'red', 'green', 'blue'][idx % 4])

    print('meshs:', meshs)

    plotter.add_axes()

    plotter.show()
    plotter.screenshot(f'{percentage}.png')
    exit(0)


    #
    # plotter.close()



    # for dfn, part in dfn_to_part.items():
    #     mesh = part['mesh_off']
    #     cur_min_bound, cur_max_bound = mesh.bound

    # plotter = pv.Plotter(off_screen=True)
    # plotter.add_mesh(dfn_to_part[1]['mesh_off'], color='white')
    # plotter.show()
    # plotter.screenshot('test.png')
    # plotter.close()
    '''