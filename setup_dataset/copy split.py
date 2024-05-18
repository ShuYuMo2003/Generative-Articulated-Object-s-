from glob import glob
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import redis
import pickle


files = glob('/mnt/d/Research/data/output/*')

selected_category = {'USB'}

shutil.rmtree('output', ignore_errors=True)
Path('output').mkdir()

_r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

for file in tqdm(files):
    info_dict = np.load(str(Path(file) / 'meta.npy'), allow_pickle=True).item()
    cate = info_dict.get('meta').get('model_cat')
    if cate in selected_category:
        shutil.copytree(file, f'output/{cate}/' + Path(file).name)
        key_name = cate + '-' + Path(file).name

        print(key_name)
        mete_dict = np.load(str(Path(file) / 'meta.npy'), allow_pickle=True).item()
        data_dict = np.load(str(Path(file) / 'data.npy'), allow_pickle=True).item()

        parts_data = []


        for part_id in data_dict.keys():
            part_data = data_dict[part_id]
            parts_data.append({
                'parent': mete_dict['mobility'][part_id]['parent'],
                'origin': np.array(part_data['from_parent']['origin'], dtype=np.float32),
                'direction': np.array(part_data['node']['direction'], dtype=np.float32),
                'bounds': np.array(part_data['node']['bounds'], dtype=np.float32),
                'trans': np.array(part_data['node']['tran'], dtype=np.float32),
            })

        current_info = {
            'meta': {
                'idx': Path(file).name,
                'cate': cate,
            },
            'part': parts_data
        }

        bytedata = pickle.dumps(current_info)
        return_type = _r.set(key_name, bytedata)
        print('pushed into and return_type = ', return_type)

        assert return_type


print('saving', _r.save())



