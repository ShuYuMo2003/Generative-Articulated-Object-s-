from pathlib import Path
import shutil
import hashlib
import random
import os

# change this before running
onet_ds_path = Path('../dataset/2_onet_sdf_dataset/result')
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

def restore():
    for fn in (onet_ds_path / 'test').glob('*.npz'):
        shutil.move(fn, onet_ds_path)
    for fn in (transfomer_ds_path / 'test').glob('*.json'):
        shutil.move(fn, transfomer_ds_path)

if __name__ == '__main__':
    # 如果test路径不空，执行restore操作
    if len(list((onet_ds_path / 'test').glob('*.npz'))) > 0 or len(list((transfomer_ds_path / 'test').glob('*.json'))) > 0:
        restore()
        print('\033[33mRestore operation finished\033[0m')
        print('onet_ds_path:', len(list(onet_ds_path.glob('*.npz'))))
        print('transfomer_ds_path:', len(list(transfomer_ds_path.glob('*.json'))))
    else:
        check_filename()
        split_onet_ds()
        split_transformer_ds()