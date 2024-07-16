import bpy

if 'io_scene_obj' not in bpy.context.preferences.addons:
    bpy.ops.preferences.addon_enable(module='io_scene_obj')

def bpy_render_meshs(obj_file_paths, output_path):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    for obj_file_path in obj_file_paths:
        bpy.ops.wm.obj_import(filepath=obj_file_path)

    bpy.context.scene.render.engine = 'CYCLES'

    bpy.ops.object.camera_add(location=(0, -10, 5), rotation=(1.1, 0, 0))
    camera = bpy.context.object
    bpy.context.scene.camera = camera

    bpy.ops.object.light_add(type='POINT', location=(5, -5, 10))

    bpy.context.scene.render.filepath = output_path

    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.resolution_percentage = 100

    bpy.ops.render.render(write_still=True)

if __name__ == '__main__':
    bpy_render_meshs(['../process_data/mesh0.obj', '../process_data/mesh1.obj'], 'output.png')