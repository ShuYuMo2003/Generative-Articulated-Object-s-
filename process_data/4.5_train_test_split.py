from pathlib import Path
import shutil
import hashlib
import os

# change this before running
onet_ds_path = Path('../dataset/2_onet_sdf_dataset')
transfomer_ds_path = Path('../dataset/4_transformer_dataset')
###

train_ratio = 0.8
train_keys = []

def str2hash(str):
    # use SHA256 to hash the string
    return int(hashlib.sha256(str.encode('utf-8')).hexdigest(), 16)

def check_filename():
    fonet_set = set()
    for fn in onet_ds_path.glob('result/*.npz'):
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

    print(f'Number of files in onet_ds_path: {len(fonet_set)}')
    print(f'Number of files in transfomer_ds_path: {len(ftrans_set)}')

    match = True
    for key in ftrans_set:
        if key not in fonet_set:
            # print(f'[ds_check] {key} not found in onet dataset')
            match = False
            break

    if not match:     
        print("\033[31m[Warning] Onet dataset does not match transformer dataset.\033[0m")

    for key in ftrans_set:
        if str2hash(key) % 100 < train_ratio * 100:
            train_keys.append(key)

def split_onet_ds():
    if not (onet_ds_path / 'result' / 'test').exists():
        os.makedirs(onet_ds_path / 'result' / 'test')

    train_count = 0
    test_count = 0

    for fn in onet_ds_path.glob('result/*.npz'):
        file_name = fn.stem
        if len(file_name.split('_')) < 2: 
            continue
        file_key = file_name.split('_')[0]+'_'+file_name.split('_')[1]
        if file_key not in train_keys:
            shutil.move(fn, transfomer_ds_path / 'test')
            test_count += 1
        else:
            train_count += 1
    
    check = True
    for fn in onet_ds_path.glob('result/*.npz'):
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

if __name__ == '__main__':
    check_filename()
    split_onet_ds()
    split_transformer_ds()