from generate_train_data import *
from glob import glob

MESH_FUSION_PATH = Path('/home/shuyumo/research/GAO/point2sdf/mesh-fusion')

selected_cate = {'USB'}
shape_paths = []
for cate in selected_cate:
    shape_paths.extend(glob(f'../setup_dataset/output/{cate}/*'))

# 把所有的 obj 文件转换成 sdf 训练集
## 拷贝所有 obj 文件
total_mesh_objs = []
out_stem_names = []
for shape in shape_paths:
    objs_path = glob(str(Path(shape) / 'mesh' / '*obj'))
    out_stem_name = []
    for obj in objs_path:
        out_stem_name.append(f'{Path(shape).stem}-{Path(obj).stem}.obj')
    total_mesh_objs.extend(objs_path)
    out_stem_names.extend(out_stem_name)

print(f'we totally have {len(total_mesh_objs)} for point-onet to overfit.')

shutil.rmtree('input_objs', ignore_errors=True)
Path('input_objs').mkdir(exist_ok=True)
for src, dst in zip(total_mesh_objs, out_stem_names):
    shutil.copy(src, f'input_objs/{dst}')

shutil.rmtree('output', ignore_errors=True)
Path('output').mkdir(exist_ok=True)
process_obj(Path('input_objs'), Path('output'))