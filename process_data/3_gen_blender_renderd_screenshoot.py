import trimesh
from rich import print
from pathlib import Path

def generate_combined_mesh_n_texture(shape_path: Path):
    pass

if __name__ == '__main__':
    raw_dataset_path = Path('../dataset/raw')
    shapes_path = raw_dataset_path.glob('*')

    for shape_path in shapes_path:
        generate_combined_mesh_n_texture(shape_path)
        break