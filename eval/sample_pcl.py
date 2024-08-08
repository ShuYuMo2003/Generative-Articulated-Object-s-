# adapted from https://github.com/JiahuiLei/NAP/

import os
import json
import trimesh
import networkx as nx
import numpy as np
from transforms3d.axangles import axangle2mat
from tqdm import tqdm
import logging
from multiprocessing import Pool
from random import shuffle
import argparse

logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)

def load_data_from_json(json_file, mesh_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    G = nx.DiGraph()
    for part in data['part']:
        mesh_path = os.path.join(mesh_dir, part['mesh'])
        mesh = trimesh.load(mesh_path)
        bbox = np.array(part['bounding_box'])
        joint_data_origin = np.array(part['joint_data_origin'])
        joint_data_direction = np.array(part['joint_data_direction'])
        limit = np.array(part['limit'])
        G.add_node(part['dfn'], mesh=mesh, bbox=bbox)
        if part['dfn_fa'] != 0:
            G.add_edge(part['dfn_fa'], part['dfn'], T_src_dst=np.eye(4), plucker=np.concatenate((joint_data_direction, joint_data_origin)), 
                       plim=limit[:2], rlim=limit[2:], src=part['dfn_fa'], dst=part['dfn'])
    
    assert nx.is_tree(G)
    return G

def sample(G, dst_fn, N_states=100, N_PCL=10000):
    mesh_list, pose_list = forward_G(G, N_frame=N_states)
    pcl_list = [sample_tmesh(mesh, N_PCL) for mesh in mesh_list]

    pose_list = np.stack(pose_list, 0)  # N_states, N_parts, 4,4
    pcl_list = np.stack(pcl_list, 0)  # N_states, N_pcl, 3
    np.savez_compressed(dst_fn, pcl=pcl_list, pose=pose_list)
    return

def sample_tmesh(tmesh, pre_sample_n):
    pcl, _ = trimesh.sample.sample_surface_even(tmesh, pre_sample_n * 2)
    while len(pcl) < pre_sample_n:
        _pcl, _ = trimesh.sample.sample_surface_even(tmesh, pre_sample_n * 2)
        pcl = np.concatenate([pcl, _pcl])
    pcl = pcl[:pre_sample_n]
    pcl = np.asarray(pcl, dtype=np.float16)
    return pcl

def screw_to_T(theta, d, l, m):
    try:
        assert abs(np.linalg.norm(l) - 1.0) < 1e-3
    except:
        print(l)
        print(np.linalg.norm(l))
        raise
    R = axangle2mat(l, theta)
    t = (np.eye(3) - R) @ (np.cross(l, m)) + l * d
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def forward_G(G, N_frame=100, mesh_key="mesh"):
    ret = []
    assert len(G.nodes) >= 2 and nx.is_tree(G)  # now only support tree viz
    # * now G is connected and acyclic
    # vid = [n for n in G.nodes]
    # v_bbox = np.stack([d["bbox"] for nid, d in G.nodes(data=True)], 0)
    # v_bbox_diff = np.abs(v_bbox[:, 1] - v_bbox[:, 0])
    # v_volume = np.prod(v_bbox_diff, axis=1)
    # try:
    #     root_vid = vid[v_volume.argmax()]
    # except:
    #     print(len(v_volume))
    #     print(len(vid))
    #     print(v_volume.argmax())
    #     raise
    root_vid = 1
    POSE, MESH = [], []
    # * sample a set of possible angle range for each joint
    for step in tqdm(range(N_frame)):
        node_traverse_list = [n for n in nx.dfs_preorder_nodes(G, root_vid)]
        T_rl_list = [np.eye(4)]  # p_root = T_rl @ p_link
        # * prepare the node pos
        for i in range(len(node_traverse_list) - 1):
            cid = node_traverse_list[i + 1]
            for e, e_data in G.edges.items():
                if cid in e:
                    # determine the direction by ensure the other end is a predessor in the traversal list
                    other_end = e[0] if e[1] == cid else e[1]
                    if node_traverse_list.index(other_end) > i:
                        continue
                    else:
                        pid = other_end

                    # T1: e_T_src_j1, T2: e_T_j2_dst
                    e_data = G.edges[e]
                    _T0 = e_data["T_src_dst"]
                    plucker = e_data["plucker"]
                    l, m = plucker[:3], plucker[3:]
                    l = l / np.linalg.norm(l)
                    plim, rlim = e_data["plim"], e_data["rlim"]

                    # ! random sample
                    # theta = np.linspace(*rlim, N_frame)[step]
                    # d = np.linspace(*plim, N_frame)[step]
                    theta = np.random.uniform(*rlim)
                    d = np.random.uniform(*plim)

                    _T1 = screw_to_T(theta, d, l, m)
                    T_src_dst = _T1 @ _T0
                    if pid == e_data["src"]:  # parent is src
                        T_parent_child = T_src_dst
                    else:  # parent is dst
                        T_parent_child = np.linalg.inv(T_src_dst)
                        # T_parent_child = T_src_dst
                    T_root_child = T_rl_list[node_traverse_list.index(pid)] @ T_parent_child
                    T_rl_list.append(T_root_child)
                    break
        assert len(T_rl_list) == len(node_traverse_list)

        # * prepare the bbox
        mesh_list = []
        for nid, T in zip(node_traverse_list, T_rl_list):
            assert mesh_key in G.nodes[nid].keys()
            mesh = G.nodes[nid][mesh_key].copy()
            mesh.apply_transform(T.copy())
            mesh_list.append(mesh)
        mesh_list = trimesh.util.concatenate(mesh_list)
        MESH.append(mesh_list)
        POSE.append(np.stack(T_rl_list, 0))

    return MESH, POSE

def thread(p):
    json_file, mesh_dir, dst_fn, N_states, N_PCL = p
    print(f"Processing {dst_fn}")
    G = load_data_from_json(json_file, mesh_dir)
    print(f"Loaded {json_file}")
    sample(G, dst_fn, N_states, N_PCL)
    print(f"Saved {dst_fn}")
    return

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run")
    arg_parser.add_argument("--info_dir", default="../dataset/1_preprocessed_info/test")
    arg_parser.add_argument("--mesh_dir", default="../dataset/1_preprocessed_mesh/test")
    arg_parser.add_argument("--dst", default="../logs/test/PCL/gt")
    # arg_parser.add_argument("--info_dir", default="../logs/test/output/1_info")
    # arg_parser.add_argument("--mesh_dir", default="../logs/test/output/2_mesh")
    # arg_parser.add_argument("--dst", default="../logs/test/PCL/gen")
    arg_parser.add_argument("--n_states", default=10, type=int)
    arg_parser.add_argument("--n_pcl", default=2048, type=int)
    arg_parser.add_argument("--n_thread", default=16, type=int)
    args = arg_parser.parse_args()

    INFO_DIR = args.info_dir
    MESH_DIR = args.mesh_dir
    DST = args.dst
    N_states = args.n_states
    N_pcl = args.n_pcl
    if os.path.exists(DST):
        for fn in os.listdir(DST):
            if fn.endswith(".npz"):
                os.remove(os.path.join(DST, fn))
    os.makedirs(DST, exist_ok=True)
    p_list = []

    for fn in os.listdir(INFO_DIR):
        if fn.endswith(".json"):
            json_file = os.path.join(INFO_DIR, fn)
            dst_fn = os.path.join(DST, fn[:-5] + ".npz")
            # if fn == "Chair_41838.json":
            #     continue
            # if fn != "Chair_2440.json":
            #     continue
            # print(f"Processing {dst_fn}")
            # thread((json_file, MESH_DIR, dst_fn, N_states, N_pcl))
            p_list.append((json_file, MESH_DIR, dst_fn, N_states, N_pcl))

    shuffle(p_list)
    with Pool(args.n_thread) as p:
        p.map(thread, p_list)
