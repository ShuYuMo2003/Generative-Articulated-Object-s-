import redis
import pickle
from rich import print
import numpy as np


_r = redis.Redis(host='localhost', port=6379, db=0)

total_parts = _r.keys()

for part in total_parts:
    idx = part.decode('utf-8').split('-')[1]
    data = pickle.loads(_r.get(part))
    for part_idx, part_data in enumerate(data['part']):
        part_data['latent'] = np.load(f'/home/shuyumo/research/GAO/point2sdf/output/generation_latent_code/latent_{idx}-mesh-{part_idx}.npy')
        print(type(part_data['latent']), part_data['latent'].shape)
    print(data)
    bytedata = pickle.dumps(data)
    return_type = _r.set(part, bytedata)
    print('pushed into and return_type = ', return_type)