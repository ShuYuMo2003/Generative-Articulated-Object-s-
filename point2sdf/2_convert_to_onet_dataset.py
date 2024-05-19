from generate_train_data import *
from glob import glob
import os
import shutil

CUURENT_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
MESH_FUSION_PATH = CUURENT_PATH / 'mesh-fusion'

selected_cate = {'USB'}
shape_paths = []
for cate in selected_cate:
    shape_paths.extend(glob(f'../setup_dataset/output/{cate}/*'))

output_mesh_path = Path('output/1_preprecessed_mesh')
output_mesh_path.mkdir(exist_ok=True, parents=True)

output_dataset_path = Path('output/2_dataset')
shutil.rmtree(output_dataset_path, ignore_errors=True)
process_obj(output_mesh_path, output_dataset_path)


# # 把所有的 obj 文件转换成 sdf 训练集
# ## 拷贝所有 obj 文件
# total_mesh_objs = []
# out_stem_names = []
# for shape in shape_paths:
#     objs_path = glob(str(Path(shape) / 'mesh' / '*obj'))
#     out_stem_name = []
#     for obj in objs_path:
#         out_stem_name.append(f'{Path(shape).stem}-{Path(obj).stem}.obj')
#     total_mesh_objs.extend(objs_path)
#     out_stem_names.extend(out_stem_name)

# print(f'we totally have {len(total_mesh_objs)} for point-onet to overfit.')

# shutil.rmtree('input_objs', ignore_errors=True)
# Path('input_objs').mkdir(exist_ok=True)
# for src, dst in zip(total_mesh_objs, out_stem_names):
#     shutil.copy(src, f'input_objs/{dst}')

# shutil.rmtree('output', ignore_errors=True)
# Path('output').mkdir(exist_ok=True)
# process_obj(Path('input_objs'), Path('output'))