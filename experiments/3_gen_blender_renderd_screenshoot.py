import trimesh
from rich import print
from pathlib import Path

def generate_combined_mesh_n_texture(shape_path: Path):
    print('Processing shape:', shape_path)
    normalized_output_path = (shape_path / 'normalized_textured_objs')
    normalized_output_path.mkdir(exist_ok=True)

    meshs = []

    for textured_obj_path in (shape_path / 'textured_objs').glob('*.obj'):
        print('[Loads]', textured_obj_path)
        mesh_or_scene = trimesh.load(textured_obj_path)
        if isinstance(mesh_or_scene, trimesh.Scene):
            for mesh in mesh_or_scene.geometry.values():
                meshs.append(mesh)
        else:
            assert isinstance(mesh_or_scene, trimesh.Trimesh)
            meshs.append(mesh_or_scene)

    print('[Info] total meshs:', len(meshs))

    combined_mesh = trimesh.util.concatenate(meshs)
    bounds = combined_mesh.bounds
    scale = 1 / max(bounds[1] - bounds[0])
    translation = -bounds[0]

    for textured_obj_path in (shape_path / 'textured_objs').glob('*.obj'):
        print('[Process]', textured_obj_path)
        mesh_or_scene = trimesh.load(textured_obj_path)
        if isinstance(mesh_or_scene, trimesh.Scene):
            for mesh in mesh_or_scene.geometry.values():
                mesh.apply_translation(translation)
                mesh.apply_scale(scale)
        else:
            assert isinstance(mesh_or_scene, trimesh.Trimesh)
            mesh.apply_translation(translation)
            mesh.apply_scale(scale)

        output_path = (normalized_output_path / textured_obj_path.name).with_suffix('.ply')
        mesh_or_scene.export(output_path)
        print('[Write]', output_path)

if __name__ == '__main__':
    raw_dataset_path = Path('../dataset/raw')
    shapes_path = raw_dataset_path.glob('*')

    for shape_path in shapes_path:
        generate_combined_mesh_n_texture(shape_path)
        break