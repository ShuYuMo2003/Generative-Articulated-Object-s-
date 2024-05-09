from glob import glob
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm


files = glob('/mnt/d/Research/data/output/*')

selected_category = {'USB'}

shutil.rmtree('output', ignore_errors=True)
Path('output').mkdir()

for file in tqdm(files):
    info_dict = np.load(str(Path(file) / 'meta.npy'), allow_pickle=True).item()
    cate = info_dict.get('meta').get('model_cat')
    if cate in selected_category:
        shutil.copytree(file, f'output/{cate}/' + Path(file).name)

