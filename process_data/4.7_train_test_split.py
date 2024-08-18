from pathlib import Path
import shutil
import hashlib
import random
import json
import os

# change this before running
dataset_root_path = Path('../dataset')
preprocessed_info_path = Path('../dataset/1_preprocessed_info')
preprocessed_mesh_path = Path('../dataset/1_preprocessed_mesh')
onet_ds_path = Path('../dataset/2_gensdf_dataset/result')
transfomer_ds_path = Path('../dataset/4_transformer_dataset')
###

train_ratio = 0.8
train_keys = []

# def str2hash(str):
#     # use SHA256 to hash the string
#     return int(hashlib.sha256(str.encode('utf-8')).hexdigest(), 16)

def check_filename():
    fonet_set = set()
    for fn in onet_ds_path.glob('*.npz'):
        file_name = fn.stem
        if len(file_name.split('_')) < 2:
            print(f'[onet_ds] {file_name} is not a valid file name')
            continue
        file_key = file_name.split('_')[0]+'_'+file_name.split('_')[1]
        fonet_set.add(file_key)
    ftrans_set = set()
    for fn in transfomer_ds_path.glob('*.json'):
        file_name = fn.stem
        if len(file_name.split('_')) < 2:
            print(f'[transformer_ds] {file_name} is not a valid file name')
            continue
        file_key = file_name.split('_')[0]+'_'+file_name.split('_')[1]
        ftrans_set.add(file_key)
    print(f'Number of keys in onet_ds_path: {len(fonet_set)}')
    print(f'Number of keys in transfomer_ds_path: {len(ftrans_set)}')
    match = True
    for key in ftrans_set:
        if key not in fonet_set:
            # print(f'[ds_check] {key} not found in onet dataset')
            match = False
            break
    if not match:
        print("\033[31m[Warning] Onet dataset does not match transformer dataset.\033[0m")
    key_list = list(ftrans_set)
    random.seed(114514)
    random.shuffle(key_list)
    train_count = int(len(key_list)*train_ratio)
    train_keys.extend(key_list[:train_count])

def split_preprocessed_info():
    if not (preprocessed_info_path / 'test').exists():
        os.makedirs(preprocessed_info_path / 'test')
    train_count = 0
    test_count = 0
    for fn in preprocessed_info_path.glob('*.json'):
        file_name = fn.stem
        if len(file_name.split('_')) < 2:
            continue
        file_key = file_name.split('_')[0]+'_'+file_name.split('_')[1]
        if file_key not in train_keys:
            shutil.move(fn, preprocessed_info_path / 'test')
            test_count += 1
        else:
            train_count += 1
    check = True
    for fn in preprocessed_info_path.glob('*.json'):
        file_name = fn.stem
        if len(file_name.split('_')) < 2:
            continue
        file_key = file_name.split('_')[0]+'_'+file_name.split('_')[1]
        if file_key not in train_keys:
            check = False
            break
    if check:
        print(f'\033[32mFinished splitting preprocessed info, Train: {train_count}, Test: {test_count}\033[0m')
    else:
        print('\033[31mError in splitting preprocessed info\033[0m')

def split_preprocessed_mesh():
    if not (preprocessed_mesh_path / 'test').exists():
        os.makedirs(preprocessed_mesh_path / 'test')
    train_count = 0
    test_count = 0
    for fn in preprocessed_mesh_path.glob('*.ply'):
        file_name = fn.stem
        if len(file_name.split('_')) < 2:
            continue
        file_key = file_name.split('_')[0]+'_'+file_name.split('_')[1]
        if file_key not in train_keys:
            shutil.move(fn, preprocessed_mesh_path / 'test')
            test_count += 1
        else:
            train_count += 1
    check = True
    for fn in preprocessed_mesh_path.glob('*.ply'):
        file_name = fn.stem
        if len(file_name.split('_')) < 2:
            continue
        file_key = file_name.split('_')[0]+'_'+file_name.split('_')[1]
        if file_key not in train_keys:
            check = False
            break
    if check:
        print(f'\033[32mFinished splitting preprocessed mesh, Train: {train_count}, Test: {test_count}\033[0m')
    else:
        print('\033[31mError in splitting preprocessed mesh\033[0m')

def split_onet_ds():
    if not (onet_ds_path / 'test').exists():
        os.makedirs(onet_ds_path / 'test')
    train_count = 0
    test_count = 0
    for fn in onet_ds_path.glob('*.npz'):
        file_name = fn.stem
        if len(file_name.split('_')) < 2:
            continue
        file_key = file_name.split('_')[0]+'_'+file_name.split('_')[1]
        if file_key not in train_keys:
            shutil.move(fn, onet_ds_path / 'test')
            # print(f"[onet] test key: {file_key}")
            test_count += 1
        else:
            train_count += 1
    check = True
    for fn in onet_ds_path.glob('*.npz'):
        file_name = fn.stem
        if len(file_name.split('_')) < 2:
            continue
        file_key = file_name.split('_')[0]+'_'+file_name.split('_')[1]
        if file_key not in train_keys:
            check = False
            break
    if check:
        print(f'\033[32mFinished splitting onet dataset, Train: {train_count}, Test: {test_count}\033[0m')
    else:
        print('\033[31mError in splitting onet dataset\033[0m')

def split_transformer_ds():
    if not (transfomer_ds_path / 'test').exists():
        os.makedirs(transfomer_ds_path / 'test')
    train_count = 0
    test_count = 0
    for fn in transfomer_ds_path.glob('*.json'):
        file_name = fn.stem
        if len(file_name.split('_')) < 2:
            continue
        file_key = file_name.split('_')[0]+'_'+file_name.split('_')[1]
        if file_key not in train_keys:
            shutil.move(fn, transfomer_ds_path / 'test')
            # print(f"[transformer] test key: {file_key}")
            test_count += 1
        else:
            train_count += 1
    check = True
    for fn in transfomer_ds_path.glob('*.json'):
        file_name = fn.stem
        if len(file_name.split('_')) < 2:
            continue
        file_key = file_name.split('_')[0]+'_'+file_name.split('_')[1]
        if file_key not in train_keys:
            check = False
            break
    if check:
        print(f'\033[32mFinished splitting transformer dataset, Train: {train_count}, Test: {test_count}\033[0m')
    else:
        print('\033[31mError in splitting transformer dataset\033[0m')

def check_empty():
    if len(list((preprocessed_info_path / 'test').glob('*.json'))) > 0:
        return False
    if len(list((preprocessed_mesh_path / 'test').glob('*.ply'))) > 0:
        return False
    if len(list((onet_ds_path / 'test').glob('*.npz'))) > 0:
        return False
    if len(list((transfomer_ds_path / 'test').glob('*.json'))) > 1:
        return False
    return True

def restore():
    for fn in (preprocessed_info_path / 'test').glob('*.json'):
        shutil.move(fn, preprocessed_info_path)
    for fn in (preprocessed_mesh_path / 'test').glob('*.ply'):
        shutil.move(fn, preprocessed_mesh_path)
    for fn in (onet_ds_path / 'test').glob('*.npz'):
        shutil.move(fn, onet_ds_path)
    for fn in (transfomer_ds_path / 'test').glob('*.json'):
        if fn.stem == 'meta':
            continue
        shutil.move(fn, transfomer_ds_path)

def save_train_keys():
    with open(dataset_root_path / 'train_keys.json', 'w') as f:
        json.dump({'train_keys': train_keys}, f)
    print(f'Train keys saved to {dataset_root_path / "train_keys.json"}')

if __name__ == '__main__':
    if not check_empty():
        restore()
        print('\033[33mRestore operation finished\033[0m')
        print('preprocessed_info_path:', len(list(preprocessed_info_path.glob('*.json'))))
        print('preprocessed_mesh_path:', len(list(preprocessed_mesh_path.glob('*.ply'))))
        print('onet_ds_path:', len(list(onet_ds_path.glob('*.npz'))))
        print('transfomer_ds_path:', len(list(transfomer_ds_path.glob('*.json'))))
    else:
        check_filename()
        split_preprocessed_info()
        split_preprocessed_mesh()
        split_onet_ds()
        split_transformer_ds()
        save_train_keys()