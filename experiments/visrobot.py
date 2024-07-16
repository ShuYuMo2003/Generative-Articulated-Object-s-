import numpy as np
from scipy.spatial.transform import Rotation as R
from urdf_parser_py.urdf import URDF
import pyvista as pv

from urdf_parser_py.urdf import Joint


# Use the SafeURDF class to parse the URDF file
def parse_urdf(urdf_path):
    return URDF.from_xml_file(urdf_path)

def get_joint_transform(joint, angle_or_distance):
    origin = joint.origin
    if origin is not None:
        pos = np.array(origin.xyz)
        rot = R.from_euler('xyz', origin.rpy).as_matrix()
    else:
        pos = np.zeros(3)
        rot = np.eye(3)

    if joint.type == 'revolute' or joint.type == 'continuous':
        axis = np.array(joint.axis)
        joint_rot = R.from_rotvec(axis * angle_or_distance).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rot @ joint_rot
        transform[:3, 3] = pos
    elif joint.type == 'prismatic':
        axis = np.array(joint.axis)
        joint_pos = axis * angle_or_distance
        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = pos + joint_pos
    else:
        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = pos

    return transform

def compute_forward_kinematics(robot, joint_angles_or_distances):
    link_poses = {}
    current_transform = np.eye(4)

    def recursive_fk(link, parent_transform):
        nonlocal current_transform
        current_transform = parent_transform

        if link in link_poses:
            return link_poses[link]

        for joint in robot.joints:
            if joint.parent == link:
                angle_or_distance = joint_angles_or_distances.get(joint.name, 0)
                joint_transform = get_joint_transform(joint, angle_or_distance)
                child_transform = parent_transform @ joint_transform

                link_poses[joint.child] = child_transform
                recursive_fk(joint.child, child_transform)

        link_poses[link] = parent_transform
        return parent_transform

    base_link = robot.get_root()
    recursive_fk(base_link, current_transform)

    return link_poses

def create_geometry_mesh(geometry, transform):
    if geometry.box:
        size = geometry.box.size
        mesh = pv.Box(bounds=(-size[0]/2, size[0]/2, -size[1]/2, size[1]/2, -size[2]/2, size[2]/2))
    elif geometry.cylinder:
        radius = geometry.cylinder.radius
        length = geometry.cylinder.length
        mesh = pv.Cylinder(center=(0, 0, -length/2), direction=(0, 0, 1), radius=radius, height=length)
    elif geometry.sphere:
        radius = geometry.sphere.radius
        mesh = pv.Sphere(radius=radius)
    elif geometry.mesh:
        # Load mesh from file (assuming STL format)
        mesh = pv.read(geometry.mesh.filename)
    else:
        mesh = None

    if mesh:
        # Apply the transformation to the mesh
        mesh.transform(transform)

    return mesh

def visualize_robot(robot, link_poses):
    plotter = pv.Plotter()

    for link_name, pose in link_poses.items():
        link = robot.link_map[link_name]
        for visual in link.visuals:
            transform = np.eye(4)
            if visual.origin:
                transform[:3, 3] = visual.origin.xyz
                transform[:3, :3] = R.from_euler('xyz', visual.origin.rpy).as_matrix()
            final_transform = pose @ transform
            mesh = create_geometry_mesh(visual.geometry, final_transform)
            if mesh:
                plotter.add_mesh(mesh, color='lightblue', opacity=0.7)

    plotter.show()

# Load URDF and set joint angles or distances
urdf_path = '../dataset/raw/2780/mobility.urdf'
robot = parse_urdf(urdf_path)
joint_angles_or_distances = {
    'joint1': np.deg2rad(30),
    'joint2': np.deg2rad(45),
    # Add more joints and their angles or distances here
}

# Compute forward kinematics
link_poses = compute_forward_kinematics(robot, joint_angles_or_distances)

# Visualize robot
visualize_robot(robot, link_poses)