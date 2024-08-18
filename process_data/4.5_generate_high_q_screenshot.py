import json
import time
import shutil
import random
import subprocess
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.append('..')
from utils.logging import console, Log
from utils import generate_random_string

script = Path('../static/blender_render_script.template.py').read_text()
bg_ply_path = Path('../static/bg.ply')
blender_main_program_path = Path('/root/blender-git/blender-4.2.0-linux-x64/blender')
n_png_per_obj = 6

random.seed(20030912)

def generate_high_q_screenshot(obj_name: str, textured_obj_path: Path, output_dir: Path):
    suffix = generate_random_string(4)
    template_path = output_dir / f'script-{obj_name}-{suffix}.py'
    log_path = output_dir / f'log-{obj_name}-{suffix}.log'

    cur_script = (script
            .replace("{{objs_path}}", textured_obj_path.as_posix())
            .replace("{{bg_ply_path}}", bg_ply_path.as_posix())
            .replace("{{output_path}}", (output_dir / (obj_name + f'-{suffix}.png')).as_posix())
            .replace("{{r}}", f'{(random.random() * (11 - 8.4) + 8.4):.5f}')
            .replace("{{azimuth}}", f'{random.randint(0, 360):.5f}')
            .replace("{{elevation}}", f'{random.randint(10, 80):.5f}')
        )

    template_path.write_text(cur_script)

    start_time = time.time()
    with open(log_path.as_posix(), 'w') as log_file:
        process = subprocess.Popen([
                blender_main_program_path.as_posix(),
                '--background',
                # '--cycles-device', 'CUDA'
                '--python', template_path.as_posix(),
            ]
            , stdout=log_file, stderr=log_file
            )
        process.wait()

    if process.returncode != 0:
        Log.critical(f'[{obj_name}] Blender failed with status {process.returncode}')
        exit(-1)
    Log.info(f'[{obj_name}] Rendered in {time.time() - start_time:.2f}s with returncode = {process.returncode}')

if __name__ == '__main__':
    raw_dataset_path = Path('../dataset/raw')
    high_q_output_path = Path('../dataset/4_screenshot_high_q')
    # shutil.rmtree(high_q_output_path, ignore_errors=True)
    high_q_output_path.mkdir(exist_ok=True)

    train_keys = json.loads((Path('../dataset') / 'train_keys.json').read_text())['train_keys']

    for raw_path in tqdm(list(raw_dataset_path.glob('*')),
                         desc='Generating High-Quality Screenshots'):
        meta = json.loads((raw_path / 'meta.json').read_text())
        obj_name = (meta['model_cat'] + '_' + meta['anno_id'])
        if obj_name not in train_keys:
            Log.info(f'[{obj_name}] Skipping')
            continue
        for _ in range(n_png_per_obj):
            generate_high_q_screenshot(obj_name, raw_path / 'textured_objs', high_q_output_path)
            Log.info(f'[{obj_name}] Done, repeat = {_ + 1}')
