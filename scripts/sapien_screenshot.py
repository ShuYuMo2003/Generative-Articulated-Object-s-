import os
from tqdm import tqdm
from requests import get
from multiprocessing import Pool
from glob import glob

url = 'https://sapien.ucsd.edu/api/data/images/IDID.png'
n_process = 8
mesh_info_ids = map(lambda x : (x.split('-')[0].split('/')[-1],
                                x.split('-')[1]),
                    glob("dataset/1_preprecessed_mesh/*-0.ply"))

image_save_path = "dataset/4_screenshot/"

if not os.path.exists(image_save_path):
    os.makedirs(image_save_path)

def download_image(mesh_info_id, path):
    with open(path, 'wb') as f:
        f.write(get(url.replace('IDID', mesh_info_id)).content)


with Pool(processes=n_process) as pool:
    result = []
    for cate, mesh_info_id in mesh_info_ids:
        result.append(pool.apply_async(download_image, args=(mesh_info_id, os.path.join(image_save_path, f"{cate}-{mesh_info_id}.png"))))

    for r in tqdm(result):
        r.get()
