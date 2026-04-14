"""
Lightweight Forward Kinematics from URDF using only stdlib xml + numpy.
No placo/pinocchio dependency.
"""

import xml.etree.ElementTree as ET

import numpy as np


def _parse_vec(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.split()])


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """Roll-Pitch-Yaw (XYZ extrinsic = ZYX intrinsic) to 3x3 rotation matrix."""
    return _rot_z(rpy[2]) @ _rot_y(rpy[1]) @ _rot_x(rpy[0])


def _make_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rpy_to_matrix(rpy)
    T[:3, 3] = xyz
    return T


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix for rotation about an arbitrary axis by angle (radians)."""
    ax = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = ax
    c, s = np.cos(angle), np.sin(angle)
    t = 1 - c
    R = np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    return T


class URDFJoint:
    def __init__(self, name: str, jtype: str, parent: str, child: str,
                 origin_xyz: np.ndarray, origin_rpy: np.ndarray, axis: np.ndarray):
        self.name = name
        self.jtype = jtype
        self.parent = parent
        self.child = child
        self.origin = _make_transform(origin_xyz, origin_rpy)
        self.axis = axis


class SimpleFK:
    """Parse URDF and compute FK for a kinematic chain ending at a given frame."""

    def __init__(self, urdf_path: str, ee_frame: str = "gripper_frame_link",
                 joint_names: list[str] | None = None):
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        # Parse all joints
        self.joints: dict[str, URDFJoint] = {}
        self.child_to_joint: dict[str, URDFJoint] = {}
        for jel in root.findall("joint"):
            name = jel.get("name")
            jtype = jel.get("type")
            parent = jel.find("parent").get("link")
            child = jel.find("child").get("link")
            origin_el = jel.find("origin")
            if origin_el is not None:
                xyz = _parse_vec(origin_el.get("xyz", "0 0 0"))
                rpy = _parse_vec(origin_el.get("rpy", "0 0 0"))
            else:
                xyz = np.zeros(3)
                rpy = np.zeros(3)
            axis_el = jel.find("axis")
            axis = _parse_vec(axis_el.get("xyz", "0 0 1")) if axis_el is not None else np.array([0, 0, 1.0])
            j = URDFJoint(name, jtype, parent, child, xyz, rpy, axis)
            self.joints[name] = j
            self.child_to_joint[child] = j

        # Build chain from base_link to ee_frame
        self.chain: list[URDFJoint] = []
        link = ee_frame
        while link in self.child_to_joint:
            j = self.child_to_joint[link]
            self.chain.insert(0, j)
            link = j.parent

        # Determine which joints in the chain are actuated (revolute)
        self.actuated_joints = [j for j in self.chain if j.jtype == "revolute"]

        # If joint_names provided, reorder/filter
        if joint_names is not None:
            name_to_idx = {j.name: i for i, j in enumerate(self.actuated_joints)}
            self.joint_order = [name_to_idx[n] for n in joint_names if n in name_to_idx]
        else:
            self.joint_order = list(range(len(self.actuated_joints)))

    def forward(self, joint_angles_deg: np.ndarray) -> np.ndarray:
        """Compute FK. Returns 4x4 transform of ee_frame.

        Args:
            joint_angles_deg: Joint angles in degrees, ordered by joint_names.

        Returns:
            4x4 homogeneous transformation matrix.
        """
        # Map input angles to actuated joints
        angles_rad = np.zeros(len(self.actuated_joints))
        for i, idx in enumerate(self.joint_order):
            if i < len(joint_angles_deg):
                angles_rad[idx] = np.deg2rad(joint_angles_deg[i])

        T = np.eye(4)
        act_idx = 0
        for j in self.chain:
            T = T @ j.origin
            if j.jtype == "revolute":
                T = T @ _axis_rotation(j.axis, angles_rad[act_idx])
                act_idx += 1

        return T

    def forward_position(self, joint_angles_deg: np.ndarray) -> np.ndarray:
        """Compute FK and return only EE position (x, y, z)."""
        return self.forward(joint_angles_deg)[:3, 3]
