import os
import bpy
import numpy as np
from mathutils import Vector

bpy.context.preferences.view.language = 'en_US'
bpy.context.preferences.view.use_translate_interface = True

resolution_x, resolution_y = (1600, 1800)
resolution_percentage = 100

PI = 3.14159265357389

class CameraParameters:
    location = Vector((2.7502, -6.9099, 4.0388))
    lens = 200.0
    sensor_width = 36
    type = 'PERSP'

    @classmethod
    def apply(cls, camera):
        camera.location = cls.location
        camera.data.lens = cls.lens
        camera.data.sensor_width = cls.sensor_width
        camera.data.type = cls.type

def focus_object(camera, obj):
    obj_center = sum((Vector(b) for b in obj.bound_box), Vector()) / 8

    direction = obj_center - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

def check_obj_bound(obj):
    vertices = [obj.matrix_world @ v.co for v in obj.data.vertices]
    min_x = min(vertices, key=lambda v: v.x).x
    max_x = max(vertices, key=lambda v: v.x).x
    min_y = min(vertices, key=lambda v: v.y).y
    max_y = max(vertices, key=lambda v: v.y).y
    min_z = min(vertices, key=lambda v: v.z).z
    max_z = max(vertices, key=lambda v: v.z).z
    return min_x, max_x, min_y, max_y, min_z, max_z

def render_shape_blender(objs_path, bg_ply_path, output_path, use_gpu):
    try: bpy.ops.object.mode_set(mode='OBJECT')
    except RuntimeError: pass
    # Delete all existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Import objects from obj path
    obj_files = [f for f in os.listdir(objs_path) if f.endswith('.obj')]
    for obj_file in obj_files:
        obj_file_path = os.path.join(objs_path, obj_file)
        bpy.ops.wm.obj_import(filepath=obj_file_path)

    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.join()
    # bpy.ops.object.shade_smooth()

    obj = bpy.context.active_object
    bpy.ops.object.mode_set(mode='EDIT')

    # First: scale
    min_x, max_x, min_y, max_y, min_z, max_z = check_obj_bound(obj)
    scale = 1.0 / max(max_x - min_x, max_y - min_y, max_z - min_z)
    obj.scale *= scale
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.transform_apply(scale=True)

    # Second: translation
    min_x, max_x, min_y, max_y, min_z, max_z = check_obj_bound(obj)
    translation = Vector((-(min_x + max_x) / 2, -(min_y + max_y) / 2, -min_z))
    obj.location += translation
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.transform_apply(location=True)

    # Load Background Mesh.
    bpy.ops.wm.ply_import(filepath=bg_ply_path)
    bpy.ops.object.shade_smooth()

    ## Add Materials To BG Mesh.
    bgmat = bpy.data.materials.new(name='bgmat')
    bgobj = bpy.context.active_object
    bgobj.data.materials.append(bgmat)
    bgmat.use_nodes = True
    edit_node = bgmat.node_tree.nodes["Principled BSDF"]

    edit_node.inputs['Base Color'].default_value = (1, 1, 1, 1) #(0.800204, 0.689428, 0, 1)
    edit_node.inputs['Metallic'].default_value = 0.444109
    edit_node.inputs['Roughness'].default_value = 0.208096
    edit_node.distribution = 'GGX'
    edit_node.inputs['Emission Strength'].default_value = 1.0

    # Add Lights
    ## TOP
    bpy.ops.object.light_add(type='POINT', location=(0, 0, 2))
    light = bpy.context.active_object
    light.data.energy = 100
    light.rotation_euler = (PI, 0, 0)

    # Add new camera
    bpy.ops.object.camera_add()
    new_cam = bpy.context.active_object
    CameraParameters.apply(new_cam)
    bpy.context.scene.camera = new_cam
    focus_object(new_cam, obj)

    # Render
    render = bpy.context.scene.render

    render.resolution_x = resolution_x
    render.resolution_y = resolution_y
    render.resolution_percentage = resolution_percentage
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 128
    render.image_settings.file_format = 'PNG'
    render.filepath = output_path

    if use_gpu:
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.preferences.addons['cycles'].preferences.get_devices()

    bpy.ops.render.render(write_still=True)


if __name__ == '__main__':
    objs_path = "../dataset/raw/2780/textured_objs"
    bg_ply_path = "bg.ply"
    output_path = "render_output.png"
    render_shape_blender(objs_path, bg_ply_path, output_path, True)

    # /root/blender-git/blender-4.2.0-linux-x64/blender --background --cycles-device CUDA --python blender_render.py
