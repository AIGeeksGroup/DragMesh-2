#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


FINGER_BODY_RE = re.compile(r"^[LR]_(Index|Middle|Pinky|Ring|Thumb)[123]$")
FINGER_JOINT_RE = re.compile(r"^[LR]_(Index|Middle|Pinky|Ring|Thumb)[123]_[xyz]$")
FINGERTIP_BODY_RE = re.compile(r"^[LR]_(Index|Middle|Pinky|Ring|Thumb)3$")
WRIST_BODY_NAMES = {"L_Wrist", "R_Wrist"}
EPS = 1e-9


@dataclass(frozen=True)
class PrimitiveComponent:
    kind: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    size: tuple[float, ...]
    mass: float
    inertia: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


def vec(values: tuple[float, ...] | list[float]) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def fmt_scalar(value: float) -> str:
    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:.10g}"


def fmt_vec(values: tuple[float, ...] | list[float]) -> str:
    return " ".join(fmt_scalar(float(v)) for v in values)


def parse_float_list(raw: str | None, length: int, default: tuple[float, ...]) -> tuple[float, ...]:
    if raw is None:
        return default
    values = tuple(float(item) for item in raw.replace(",", " ").split())
    if len(values) != length:
        raise ValueError(f"Expected {length} values, got {len(values)} from {raw!r}")
    return values


def is_finger_body(name: str | None) -> bool:
    return bool(name and FINGER_BODY_RE.match(name))


def is_finger_joint(name: str | None) -> bool:
    return bool(name and FINGER_JOINT_RE.match(name))


def add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(a: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a: tuple[float, float, float]) -> float:
    return math.sqrt(dot(a, a))


def normalize(a: tuple[float, float, float]) -> tuple[float, float, float]:
    length = norm(a)
    if length < EPS:
        return (0.0, 0.0, 1.0)
    return (a[0] / length, a[1] / length, a[2] / length)


def identity_matrix() -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    return (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )


def transpose(matrix):
    return tuple(zip(*matrix))


def matmul(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def matvec(matrix, vector):
    return tuple(sum(matrix[i][j] * vector[j] for j in range(3)) for i in range(3))


def outer(v):
    return (
        (v[0] * v[0], v[0] * v[1], v[0] * v[2]),
        (v[1] * v[0], v[1] * v[1], v[1] * v[2]),
        (v[2] * v[0], v[2] * v[1], v[2] * v[2]),
    )


def add_matrix(a, b):
    return tuple(
        tuple(a[i][j] + b[i][j] for j in range(3))
        for i in range(3)
    )


def scale_matrix(a, s: float):
    return tuple(
        tuple(a[i][j] * s for j in range(3))
        for i in range(3)
    )


def rotate_inertia(inertia, rotation):
    return matmul(matmul(rotation, inertia), transpose(rotation))


def shift_inertia(inertia, mass: float, displacement: tuple[float, float, float]):
    displacement_sq = dot(displacement, displacement)
    offset = tuple(
        tuple(
            mass * ((displacement_sq if i == j else 0.0) - outer(displacement)[i][j])
            for j in range(3)
        )
        for i in range(3)
    )
    return add_matrix(inertia, offset)


def quat_wxyz_to_matrix(quat):
    w, x, y, z = quat
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < EPS:
        return identity_matrix()
    w, x, y, z = w / n, x / n, y / n, z / n
    return (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ),
        (
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ),
        (
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )


def matrix_to_rpy(rotation):
    sy = max(-1.0, min(1.0, -rotation[2][0]))
    pitch = math.asin(sy)
    if abs(abs(sy) - 1.0) < 1e-8:
        roll = math.atan2(-rotation[0][1], rotation[1][1])
        yaw = 0.0
    else:
        roll = math.atan2(rotation[2][1], rotation[2][2])
        yaw = math.atan2(rotation[1][0], rotation[0][0])
    return (roll, pitch, yaw)


def rotation_from_z(direction: tuple[float, float, float]):
    z_axis = normalize(direction)
    helper = (1.0, 0.0, 0.0) if abs(z_axis[0]) < 0.9 else (0.0, 1.0, 0.0)
    x_axis = normalize(cross(helper, z_axis))
    y_axis = cross(z_axis, x_axis)
    return (
        (x_axis[0], y_axis[0], z_axis[0]),
        (x_axis[1], y_axis[1], z_axis[1]),
        (x_axis[2], y_axis[2], z_axis[2]),
    )


def box_inertia(mass: float, size_xyz: tuple[float, float, float]):
    x, y, z = size_xyz
    return (
        (mass * (y * y + z * z) / 12.0, 0.0, 0.0),
        (0.0, mass * (x * x + z * z) / 12.0, 0.0),
        (0.0, 0.0, mass * (x * x + y * y) / 12.0),
    )


def cylinder_inertia(mass: float, radius: float, length: float):
    radial = mass * (3.0 * radius * radius + length * length) / 12.0
    axial = 0.5 * mass * radius * radius
    return (
        (radial, 0.0, 0.0),
        (0.0, radial, 0.0),
        (0.0, 0.0, axial),
    )


def sphere_inertia(mass: float, radius: float):
    diag = 0.4 * mass * radius * radius
    return (
        (diag, 0.0, 0.0),
        (0.0, diag, 0.0),
        (0.0, 0.0, diag),
    )


def tiny_inertial_values():
    return {
        "mass": 1e-6,
        "com": (0.0, 0.0, 0.0),
        "inertia": (
            (1e-9, 0.0, 0.0),
            (0.0, 1e-9, 0.0),
            (0.0, 0.0, 1e-9),
        ),
    }


def determine_joint_limits(joint_name: str) -> tuple[float, float]:
    if re.search(r"_(Index|Middle|Pinky|Ring|Thumb)[123]_[xyz]$", joint_name):
        return (10.0, 5.0)
    if re.search(r"_(Elbow|Wrist)_[xyz]$", joint_name):
        return (300.0, 100.0)
    if re.search(r"(Torso|Spine|Chest)_[xyz]$", joint_name):
        return (500.0, 100.0)
    if re.search(r"(Neck|Head|.*_Thorax|.*_Shoulder)_[xyz]$", joint_name):
        return (500.0, 100.0)
    if re.search(r"_(Hip|Knee|Ankle)_[xyz]$", joint_name):
        return (500.0, 100.0)
    if re.search(r"_Toe_[xyz]$", joint_name):
        return (500.0, 100.0)
    return (500.0, 100.0)


def primitive_from_box(geom) -> list[PrimitiveComponent]:
    density = float(geom.get("density", "1000"))
    half_size = parse_float_list(geom.get("size"), 3, (0.01, 0.01, 0.01))
    full_size = tuple(2.0 * value for value in half_size)
    volume = full_size[0] * full_size[1] * full_size[2]
    mass = density * volume
    position = parse_float_list(geom.get("pos"), 3, (0.0, 0.0, 0.0))
    rotation = quat_wxyz_to_matrix(parse_float_list(geom.get("quat"), 4, (1.0, 0.0, 0.0, 0.0)))
    inertia = rotate_inertia(box_inertia(mass, vec(full_size)), rotation)
    return [
        PrimitiveComponent(
            kind="box",
            xyz=vec(position),
            rpy=matrix_to_rpy(rotation),
            size=full_size,
            mass=mass,
            inertia=inertia,
        )
    ]


def primitive_from_capsule(geom) -> list[PrimitiveComponent]:
    density = float(geom.get("density", "1000"))
    size_values = parse_float_list(geom.get("size"), 1, (0.01,))
    radius = size_values[0]
    fromto = parse_float_list(geom.get("fromto"), 6, (0.0, 0.0, -0.01, 0.0, 0.0, 0.01))
    point_a = (fromto[0], fromto[1], fromto[2])
    point_b = (fromto[3], fromto[4], fromto[5])
    axis = sub(point_b, point_a)
    length = norm(axis)
    if length < EPS:
        mass = density * (4.0 / 3.0) * math.pi * radius ** 3
        inertia = sphere_inertia(mass, radius)
        midpoint = scale(add(point_a, point_b), 0.5)
        return [
            PrimitiveComponent(
                kind="sphere",
                xyz=midpoint,
                rpy=(0.0, 0.0, 0.0),
                size=(radius,),
                mass=mass,
                inertia=inertia,
            )
        ]

    rotation = rotation_from_z(axis)
    cylinder_mass = density * math.pi * radius * radius * length
    sphere_mass_each = density * (2.0 / 3.0) * math.pi * radius ** 3
    midpoint = scale(add(point_a, point_b), 0.5)
    return [
        PrimitiveComponent(
            kind="cylinder",
            xyz=midpoint,
            rpy=matrix_to_rpy(rotation),
            size=(radius, length),
            mass=cylinder_mass,
            inertia=rotate_inertia(cylinder_inertia(cylinder_mass, radius, length), rotation),
        ),
        PrimitiveComponent(
            kind="sphere",
            xyz=point_a,
            rpy=(0.0, 0.0, 0.0),
            size=(radius,),
            mass=sphere_mass_each,
            inertia=sphere_inertia(sphere_mass_each, radius),
        ),
        PrimitiveComponent(
            kind="sphere",
            xyz=point_b,
            rpy=(0.0, 0.0, 0.0),
            size=(radius,),
            mass=sphere_mass_each,
            inertia=sphere_inertia(sphere_mass_each, radius),
        ),
    ]


def body_primitives(body) -> list[PrimitiveComponent]:
    components: list[PrimitiveComponent] = []
    for geom in body.findall("geom"):
        geom_type = geom.get("type", "sphere")
        if geom_type == "plane":
            continue
        if geom_type == "box":
            components.extend(primitive_from_box(geom))
        elif geom_type == "capsule":
            components.extend(primitive_from_capsule(geom))
        else:
            raise ValueError(f"Unsupported geom type for URDF export: {geom_type}")
    return components


def aggregate_inertial(components: list[PrimitiveComponent]):
    if not components:
        return tiny_inertial_values()

    total_mass = sum(component.mass for component in components)
    if total_mass < EPS:
        return tiny_inertial_values()

    weighted_com = (0.0, 0.0, 0.0)
    for component in components:
        weighted_com = add(weighted_com, scale(component.xyz, component.mass))
    center_of_mass = scale(weighted_com, 1.0 / total_mass)

    inertia_at_com = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    for component in components:
        displacement = sub(component.xyz, center_of_mass)
        inertia_at_com = add_matrix(
            inertia_at_com,
            shift_inertia(component.inertia, component.mass, displacement),
        )

    return {
        "mass": total_mass,
        "com": center_of_mass,
        "inertia": inertia_at_com,
    }


def add_origin(parent, xyz, rpy):
    ET.SubElement(parent, "origin", xyz=fmt_vec(xyz), rpy=fmt_vec(rpy))


def add_inertial(link, components: list[PrimitiveComponent] | None = None, dummy: bool = False):
    values = tiny_inertial_values() if dummy or not components else aggregate_inertial(components)
    inertial = ET.SubElement(link, "inertial")
    add_origin(inertial, values["com"], (0.0, 0.0, 0.0))
    ET.SubElement(inertial, "mass", value=fmt_scalar(values["mass"]))
    inertia = values["inertia"]
    ET.SubElement(
        inertial,
        "inertia",
        ixx=fmt_scalar(inertia[0][0]),
        ixy=fmt_scalar(inertia[0][1]),
        ixz=fmt_scalar(inertia[0][2]),
        iyy=fmt_scalar(inertia[1][1]),
        iyz=fmt_scalar(inertia[1][2]),
        izz=fmt_scalar(inertia[2][2]),
    )


def add_geometry(parent, component: PrimitiveComponent):
    geometry = ET.SubElement(parent, "geometry")
    if component.kind == "box":
        ET.SubElement(geometry, "box", size=fmt_vec(component.size))
    elif component.kind == "cylinder":
        ET.SubElement(
            geometry,
            "cylinder",
            radius=fmt_scalar(component.size[0]),
            length=fmt_scalar(component.size[1]),
        )
    elif component.kind == "sphere":
        ET.SubElement(geometry, "sphere", radius=fmt_scalar(component.size[0]))
    else:
        raise ValueError(f"Unsupported primitive kind: {component.kind}")


def add_visual_and_collision(link, components: list[PrimitiveComponent]):
    for component in components:
        visual = ET.SubElement(link, "visual")
        add_origin(visual, component.xyz, component.rpy)
        add_geometry(visual, component)
        material = ET.SubElement(visual, "material", name="smplx_body")
        ET.SubElement(material, "color", rgba="0.8 0.6 0.4 1")

        collision = ET.SubElement(link, "collision")
        add_origin(collision, component.xyz, component.rpy)
        add_geometry(collision, component)


def create_link(robot, name: str, components: list[PrimitiveComponent] | None = None, dummy: bool = False):
    link = ET.SubElement(robot, "link", name=name)
    add_inertial(link, components, dummy=dummy)
    if not dummy and components:
        add_visual_and_collision(link, components)
    return link


def joint_origin(body, is_first_joint: bool):
    if is_first_joint:
        xyz = parse_float_list(body.get("pos"), 3, (0.0, 0.0, 0.0))
        quat = parse_float_list(body.get("quat"), 4, (1.0, 0.0, 0.0, 0.0))
        rpy = matrix_to_rpy(quat_wxyz_to_matrix(quat))
        return (vec(xyz), rpy)
    return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def add_joint(robot, *, name: str, joint_type: str, parent: str, child: str, xyz, rpy, axis=None, limit=None, damping: float | None = None):
    joint = ET.SubElement(robot, "joint", name=name, type=joint_type)
    add_origin(joint, xyz, rpy)
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    if axis is not None and joint_type not in {"fixed", "floating"}:
        ET.SubElement(joint, "axis", xyz=fmt_vec(axis))
    if limit is not None and joint_type == "revolute":
        ET.SubElement(
            joint,
            "limit",
            lower=fmt_scalar(limit["lower"]),
            upper=fmt_scalar(limit["upper"]),
            effort=fmt_scalar(limit["effort"]),
            velocity=fmt_scalar(limit["velocity"]),
        )
    if damping is not None and joint_type in {"revolute", "continuous"}:
        ET.SubElement(joint, "dynamics", damping=fmt_scalar(damping), friction="0")
    return joint


def collect_subtree_names(body):
    body_names = []
    joint_names = []
    for node in body.iter("body"):
        body_name = node.get("name")
        if body_name:
            body_names.append(body_name)
        for joint in node.findall("joint"):
            joint_name = joint.get("name")
            joint_type = joint.get("type")
            if joint_name and joint_type != "free":
                joint_names.append(joint_name)
    return body_names, joint_names


def find_body_by_name(root, body_name: str):
    return root.find(f".//body[@name='{body_name}']")


def make_hand_only_tree(source_root, wrist_body_name: str, robot_name: str):
    wrist_body = find_body_by_name(source_root, wrist_body_name)
    if wrist_body is None:
        raise ValueError(f"Could not find wrist body {wrist_body_name}")

    root = ET.Element("mujoco", model=robot_name)
    compiler = source_root.find("compiler")
    if compiler is not None:
        root.append(copy.deepcopy(compiler))

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        pos="0 0 2",
        dir="0 0 -1",
        diffuse="1 1 1",
        specular="0.1 0.1 0.1",
        directional="true",
    )

    base_name = f"{wrist_body_name}_Base"
    base_body = ET.SubElement(worldbody, "body", name=base_name, pos="0 0 0")
    ET.SubElement(
        base_body,
        "joint",
        name="floating_base_joint",
        type="free",
        limited="false",
        actuatorfrclimited="false",
    )

    hand_body = copy.deepcopy(wrist_body)
    hand_body.set("pos", "0 0 0")
    hand_body.set("quat", "1 0 0 0")
    for child in list(hand_body.findall("body")):
        hand_body.remove(child)
    base_body.append(hand_body)

    wrist_joint_names = [
        joint.get("name")
        for joint in hand_body.findall("joint")
        if joint.get("name") and joint.get("type") != "free"
    ]

    actuator = ET.SubElement(root, "actuator")
    source_actuator = source_root.find("actuator")
    if source_actuator is not None:
        copied_any = False
        for motor in source_actuator:
            if motor.get("joint") in wrist_joint_names:
                actuator.append(copy.deepcopy(motor))
                copied_any = True
        if not copied_any:
            for joint_name in wrist_joint_names:
                ET.SubElement(actuator, "motor", name=joint_name, joint=joint_name, gear="500")

    ET.SubElement(root, "contact")
    ET.SubElement(root, "sensor")
    ET.SubElement(root, "size", njmax="64", nconmax="64")
    return root


def make_full_hand_tree(source_root, wrist_body_name: str, robot_name: str):
    wrist_body = find_body_by_name(source_root, wrist_body_name)
    if wrist_body is None:
        raise ValueError(f"Could not find wrist body {wrist_body_name}")

    root = ET.Element("mujoco", model=robot_name)
    compiler = source_root.find("compiler")
    if compiler is not None:
        root.append(copy.deepcopy(compiler))

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        pos="0 0 2",
        dir="0 0 -1",
        diffuse="1 1 1",
        specular="0.1 0.1 0.1",
        directional="true",
    )

    base_name = f"{wrist_body_name}_Base"
    base_body = ET.SubElement(worldbody, "body", name=base_name, pos="0 0 0")
    ET.SubElement(
        base_body,
        "joint",
        name="floating_base_joint",
        type="free",
        limited="false",
        actuatorfrclimited="false",
    )

    hand_body = copy.deepcopy(wrist_body)
    hand_body.set("pos", "0 0 0")
    hand_body.set("quat", "1 0 0 0")
    base_body.append(hand_body)

    hand_joint_names = [
        joint.get("name")
        for joint in hand_body.iter("joint")
        if joint.get("name") and joint.get("type") != "free"
    ]

    actuator = ET.SubElement(root, "actuator")
    source_actuator = source_root.find("actuator")
    if source_actuator is not None:
        copied_any = False
        for motor in source_actuator:
            if motor.get("joint") in hand_joint_names:
                actuator.append(copy.deepcopy(motor))
                copied_any = True
        if not copied_any:
            for joint_name in hand_joint_names:
                ET.SubElement(actuator, "motor", name=joint_name, joint=joint_name, gear="500")

    ET.SubElement(root, "contact")
    ET.SubElement(root, "sensor")
    ET.SubElement(root, "size", njmax="256", nconmax="256")
    return root


def make_palms_only_tree(source_root):
    root = copy.deepcopy(source_root)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no worldbody")
    root_body = worldbody.find("body")
    if root_body is None:
        raise ValueError("MJCF has no root body")

    removed_body_names: list[str] = []
    removed_joint_names: list[str] = []

    def strip_fingers(body):
        for child in list(body.findall("body")):
            child_name = child.get("name")
            if is_finger_body(child_name):
                body_names, joint_names = collect_subtree_names(child)
                removed_body_names.extend(body_names)
                removed_joint_names.extend(joint_names)
                body.remove(child)
            else:
                strip_fingers(child)

    strip_fingers(root_body)

    actuator = root.find("actuator")
    if actuator is not None:
        for motor in list(actuator):
            if motor.get("joint") in removed_joint_names:
                actuator.remove(motor)

    contact = root.find("contact")
    if contact is not None:
        for exclude in list(contact):
            if exclude.get("body1") in removed_body_names or exclude.get("body2") in removed_body_names:
                contact.remove(exclude)

    return root, sorted(set(removed_body_names)), sorted(set(removed_joint_names))


def build_urdf(source_root, *, robot_name: str, angle_in_radians: bool):
    robot = ET.Element("robot", name=robot_name)
    ET.SubElement(robot, "material", name="smplx_body")

    joint_records = []
    body_records = []

    worldbody = source_root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no worldbody")
    root_body = worldbody.find("body")
    if root_body is None:
        raise ValueError("MJCF has no root body")

    angle_scale = 1.0 if angle_in_radians else math.pi / 180.0

    def walk(body, parent_link_name: str | None):
        body_name = body.get("name")
        if not body_name:
            raise ValueError("All bodies must be named")

        articulated_joints = [joint for joint in body.findall("joint") if joint.get("type") != "free"]
        components = body_primitives(body)

        if parent_link_name is None:
            create_link(robot, body_name, components=components, dummy=False)
            body_records.append(
                {
                    "name": body_name,
                    "parent": None,
                    "joint_names": [],
                    "geom_count": len(body.findall("geom")),
                    "primitive_count": len(components),
                }
            )
        elif not articulated_joints:
            create_link(robot, body_name, components=components, dummy=False)
            xyz, rpy = joint_origin(body, True)
            add_joint(
                robot,
                name=f"{body_name}__fixed",
                joint_type="fixed",
                parent=parent_link_name,
                child=body_name,
                xyz=xyz,
                rpy=rpy,
            )
            joint_records.append(
                {
                    "name": f"{body_name}__fixed",
                    "type": "fixed",
                    "parent_link": parent_link_name,
                    "child_link": body_name,
                }
            )
            body_records.append(
                {
                    "name": body_name,
                    "parent": parent_link_name,
                    "joint_names": [],
                    "geom_count": len(body.findall("geom")),
                    "primitive_count": len(components),
                }
            )
        else:
            current_parent = parent_link_name
            joint_names_for_body = []
            for idx, joint in enumerate(articulated_joints):
                joint_name = joint.get("name")
                if not joint_name:
                    raise ValueError(f"Body {body_name} has an unnamed joint")
                child_link = body_name if idx == len(articulated_joints) - 1 else f"{body_name}__{idx}"
                if idx == len(articulated_joints) - 1:
                    create_link(robot, child_link, components=components, dummy=False)
                else:
                    create_link(robot, child_link, components=None, dummy=True)

                xyz, rpy = joint_origin(body, idx == 0)
                axis = parse_float_list(joint.get("axis"), 3, (1.0, 0.0, 0.0))
                normalized_axis = normalize(vec(axis))
                limited = joint.get("limited", "true").lower() != "false"
                joint_range = parse_float_list(joint.get("range"), 2, (0.0, 0.0))
                effort, velocity = determine_joint_limits(joint_name)
                limit = None
                joint_type = "revolute"
                if limited:
                    limit = {
                        "lower": joint_range[0] * angle_scale,
                        "upper": joint_range[1] * angle_scale,
                        "effort": effort,
                        "velocity": velocity,
                    }
                else:
                    joint_type = "continuous"
                damping = float(joint.get("damping", "0"))
                add_joint(
                    robot,
                    name=joint_name,
                    joint_type=joint_type,
                    parent=current_parent,
                    child=child_link,
                    xyz=xyz,
                    rpy=rpy,
                    axis=normalized_axis,
                    limit=limit,
                    damping=damping,
                )
                joint_names_for_body.append(joint_name)
                joint_records.append(
                    {
                        "name": joint_name,
                        "type": joint_type,
                        "parent_link": current_parent,
                        "child_link": child_link,
                        "axis": list(normalized_axis),
                        "origin_xyz": list(xyz),
                        "origin_rpy": list(rpy),
                        "limit": limit,
                        "damping": damping,
                    }
                )
                current_parent = child_link

            body_records.append(
                {
                    "name": body_name,
                    "parent": parent_link_name,
                    "joint_names": joint_names_for_body,
                    "geom_count": len(body.findall("geom")),
                    "primitive_count": len(components),
                }
            )

        for child_body in body.findall("body"):
            walk(child_body, body_name)

    walk(root_body, None)
    return ET.ElementTree(robot), body_records, joint_records


def validate_urdf(path: Path):
    tree = ET.parse(path)
    root = tree.getroot()
    links = [link.get("name") for link in root.findall("link")]
    joints = [joint.get("name") for joint in root.findall("joint")]
    if len(links) != len(set(links)):
        raise ValueError(f"Duplicate URDF links found in {path}")
    if len(joints) != len(set(joints)):
        raise ValueError(f"Duplicate URDF joints found in {path}")
    link_set = set(links)
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"Joint missing parent/child in {path}")
        if parent.get("link") not in link_set or child.get("link") not in link_set:
            raise ValueError(f"Joint references unknown link in {path}: {joint.get('name')}")


def validate_palms_only_mjcf(path: Path):
    tree = ET.parse(path)
    root = tree.getroot()
    for body in root.findall(".//body"):
        if is_finger_body(body.get("name")):
            raise ValueError(f"Finger body still present in palms-only MJCF: {body.get('name')}")
    joint_names = {
        joint.get("name")
        for joint in root.findall(".//joint")
        if joint.get("name") and joint.get("type") != "free"
    }
    for motor in root.findall(".//actuator/*"):
        target_joint = motor.get("joint")
        if target_joint and target_joint not in joint_names:
            raise ValueError(f"Actuator references missing joint in palms-only MJCF: {target_joint}")


def validate_hand_only_mjcf(path: Path, wrist_body_name: str):
    tree = ET.parse(path)
    root = tree.getroot()
    body_names = [body.get("name") for body in root.findall(".//body") if body.get("name")]
    if wrist_body_name not in body_names:
        raise ValueError(f"Standalone hand MJCF missing wrist body {wrist_body_name}")
    for body_name in body_names:
        if is_finger_body(body_name):
            raise ValueError(f"Finger body still present in hand-only MJCF: {body_name}")
    joint_names = {
        joint.get("name")
        for joint in root.findall(".//joint")
        if joint.get("name")
    }
    for motor in root.findall(".//actuator/*"):
        target_joint = motor.get("joint")
        if target_joint and target_joint not in joint_names:
            raise ValueError(f"Actuator references missing joint in hand-only MJCF: {target_joint}")


def validate_full_hand_mjcf(path: Path, wrist_body_name: str):
    tree = ET.parse(path)
    root = tree.getroot()
    body_names = [body.get("name") for body in root.findall(".//body") if body.get("name")]
    if wrist_body_name not in body_names:
        raise ValueError(f"Standalone hand MJCF missing wrist body {wrist_body_name}")
    finger_bodies = [name for name in body_names if is_finger_body(name)]
    if not finger_bodies:
        raise ValueError(f"Standalone full hand MJCF is missing finger bodies for {wrist_body_name}")
    joint_names = {
        joint.get("name")
        for joint in root.findall(".//joint")
        if joint.get("name")
    }
    for motor in root.findall(".//actuator/*"):
        target_joint = motor.get("joint")
        if target_joint and target_joint not in joint_names:
            raise ValueError(f"Actuator references missing joint in full hand MJCF: {target_joint}")


def write_xml(tree: ET.ElementTree, path: Path):
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def compiler_uses_radians(root) -> bool:
    compiler = root.find("compiler")
    if compiler is None:
        return False
    return compiler.get("angle", "degree").lower() == "radian"


def variant_body_groups(body_names: list[str], joint_names: list[str], palms_only: bool):
    wrist_bodies = [name for name in body_names if name in WRIST_BODY_NAMES]
    fingertip_bodies = [name for name in body_names if FINGERTIP_BODY_RE.match(name)]
    finger_bodies = [name for name in body_names if is_finger_body(name)]
    if palms_only:
        contact_bodies = wrist_bodies
    else:
        contact_bodies = finger_bodies
    return {
        "wrist_bodies": wrist_bodies,
        "finger_bodies": finger_bodies,
        "fingertip_bodies": fingertip_bodies,
        "contact_bodies": contact_bodies,
        "finger_joint_names": [name for name in joint_names if is_finger_joint(name)],
    }


def relpath(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def write_metadata(path: Path, *, workspace_root: Path, repo_root: Path, variant_name: str, robot_name: str, mjcf_path: Path, urdf_path: Path, body_records, joint_records, removed_body_names=None, removed_joint_names=None, palms_only: bool = False):
    body_names = [record["name"] for record in body_records]
    joint_names = [record["name"] for record in joint_records if not record["name"].endswith("__fixed")]
    groups = variant_body_groups(body_names, joint_names, palms_only=palms_only)
    metadata = {
        "variant_name": variant_name,
        "robot_name": robot_name,
        "paths": {
            "mjcf": relpath(mjcf_path, workspace_root),
            "urdf": relpath(urdf_path, workspace_root),
            "source_mjcf": relpath(repo_root / "protomotions" / "data" / "assets" / "mjcf" / "smplx_humanoid.xml", workspace_root),
            "source_usd": relpath(repo_root / "protomotions" / "data" / "assets" / "usd" / "smplx_humanoid.usda", workspace_root),
        },
        "protomotions_asset_config": {
            "asset_root": relpath(path.parent, workspace_root),
            "asset_file_name": mjcf_path.name,
            "fix_base_link": False,
            "self_collisions": True,
        },
        "gym_loader_hints": {
            "asset_root": relpath(path.parent, workspace_root),
            "asset_file_name": urdf_path.name,
            "fix_base_link": False,
            "collapse_fixed_joints": False,
        },
        "counts": {
            "body_count": len(body_names),
            "articulated_joint_count": len(joint_names),
            "urdf_joint_count": len(joint_records),
        },
        "groups": groups,
        "removed_body_names": removed_body_names or [],
        "removed_joint_names": removed_joint_names or [],
        "body_records": body_records,
        "joint_records": joint_records,
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    workspace_root = Path(__file__).resolve().parents[1]
    repo_root = workspace_root / "ProtoMotions"
    out_root = workspace_root / "smplx_variants"
    full_dir = out_root / "full_fingers"
    hands_dir = out_root / "hands_only"
    palms_dir = out_root / "palms_only"
    full_dir.mkdir(parents=True, exist_ok=True)
    hands_dir.mkdir(parents=True, exist_ok=True)
    palms_dir.mkdir(parents=True, exist_ok=True)

    source_mjcf_path = repo_root / "protomotions" / "data" / "assets" / "mjcf" / "smplx_humanoid.xml"
    source_tree = ET.parse(source_mjcf_path)
    source_root = source_tree.getroot()
    angle_in_radians = compiler_uses_radians(source_root)

    full_mjcf_path = full_dir / "smplx_full_fingers.xml"
    full_urdf_path = full_dir / "smplx_full_fingers.urdf"
    full_metadata_path = full_dir / "metadata.json"
    hands_metadata_path = hands_dir / "metadata.json"
    palms_metadata_path = palms_dir / "metadata.json"
    overview_path = out_root / "overview.json"

    shutil.copyfile(source_mjcf_path, full_mjcf_path)

    palms_only_root, removed_body_names, removed_joint_names = make_palms_only_tree(source_root)

    full_urdf_tree, full_body_records, full_joint_records = build_urdf(
        source_root,
        robot_name="smplx_full_fingers",
        angle_in_radians=angle_in_radians,
    )
    write_xml(full_urdf_tree, full_urdf_path)
    validate_urdf(full_urdf_path)

    full_metadata = write_metadata(
        full_metadata_path,
        workspace_root=workspace_root,
        repo_root=repo_root,
        variant_name="full_fingers",
        robot_name="smplx_full_fingers",
        mjcf_path=full_mjcf_path,
        urdf_path=full_urdf_path,
        body_records=full_body_records,
        joint_records=full_joint_records,
        palms_only=False,
    )

    standalone_full_hands = {}
    hand_specs = [
        ("left", "L_Wrist", "smplx_left_hand"),
        ("right", "R_Wrist", "smplx_right_hand"),
    ]
    for hand_key, wrist_body_name, robot_name in hand_specs:
        hand_mjcf_root = make_full_hand_tree(source_root, wrist_body_name, robot_name)
        hand_mjcf_path = hands_dir / f"{robot_name}.xml"
        hand_urdf_path = hands_dir / f"{robot_name}.urdf"
        hand_metadata_path = hands_dir / f"{robot_name}.json"

        write_xml(ET.ElementTree(hand_mjcf_root), hand_mjcf_path)
        validate_full_hand_mjcf(hand_mjcf_path, wrist_body_name)

        hand_urdf_tree, hand_body_records, hand_joint_records = build_urdf(
            hand_mjcf_root,
            robot_name=robot_name,
            angle_in_radians=angle_in_radians,
        )
        write_xml(hand_urdf_tree, hand_urdf_path)
        validate_urdf(hand_urdf_path)

        standalone_full_hands[hand_key] = write_metadata(
            hand_metadata_path,
            workspace_root=workspace_root,
            repo_root=repo_root,
            variant_name=f"{hand_key}_hand",
            robot_name=robot_name,
            mjcf_path=hand_mjcf_path,
            urdf_path=hand_urdf_path,
            body_records=hand_body_records,
            joint_records=hand_joint_records,
            palms_only=False,
        )

    hands_metadata = {
        "variant_name": "hands_only",
        "description": "Standalone full-hand assets that preserve the wrist, palm, and all finger bodies for each side.",
        "assets": {
            hand_key: {
                "mjcf": metadata["paths"]["mjcf"],
                "urdf": metadata["paths"]["urdf"],
                "metadata": relpath(hands_dir / f"{metadata['robot_name']}.json", workspace_root),
                "counts": metadata["counts"],
                "groups": metadata["groups"],
            }
            for hand_key, metadata in standalone_full_hands.items()
        },
    }
    hands_metadata_path.write_text(json.dumps(hands_metadata, indent=2), encoding="utf-8")

    standalone_palms = {}
    palm_specs = [
        ("left", "L_Wrist", "smplx_left_palm_wrist"),
        ("right", "R_Wrist", "smplx_right_palm_wrist"),
    ]
    for hand_key, wrist_body_name, robot_name in palm_specs:
        hand_mjcf_root = make_hand_only_tree(source_root, wrist_body_name, robot_name)
        hand_mjcf_path = palms_dir / f"{robot_name}.xml"
        hand_urdf_path = palms_dir / f"{robot_name}.urdf"
        hand_metadata_path = palms_dir / f"{robot_name}.json"

        write_xml(ET.ElementTree(hand_mjcf_root), hand_mjcf_path)
        validate_hand_only_mjcf(hand_mjcf_path, wrist_body_name)

        hand_urdf_tree, hand_body_records, hand_joint_records = build_urdf(
            hand_mjcf_root,
            robot_name=robot_name,
            angle_in_radians=angle_in_radians,
        )
        write_xml(hand_urdf_tree, hand_urdf_path)
        validate_urdf(hand_urdf_path)

        standalone_palms[hand_key] = write_metadata(
            hand_metadata_path,
            workspace_root=workspace_root,
            repo_root=repo_root,
            variant_name=f"{hand_key}_palm_wrist",
            robot_name=robot_name,
            mjcf_path=hand_mjcf_path,
            urdf_path=hand_urdf_path,
            body_records=hand_body_records,
            joint_records=hand_joint_records,
            palms_only=True,
        )

    palms_metadata = {
        "variant_name": "palms_only",
        "description": "Standalone hand assets that preserve only the wrist and palm body for each side.",
        "assets": {
            hand_key: {
                "mjcf": metadata["paths"]["mjcf"],
                "urdf": metadata["paths"]["urdf"],
                "metadata": relpath(palms_dir / f"{metadata['robot_name']}.json", workspace_root),
                "counts": metadata["counts"],
                "groups": metadata["groups"],
            }
            for hand_key, metadata in standalone_palms.items()
        },
        "removed_from_original_full_body": {
            "removed_body_names": removed_body_names,
            "removed_joint_names": removed_joint_names,
        },
    }
    palms_metadata_path.write_text(json.dumps(palms_metadata, indent=2), encoding="utf-8")

    overview = {
        "generator": "smplx_variants/generate_smplx_variants.py",
        "source_mjcf": relpath(source_mjcf_path, workspace_root),
        "variants": {
            "full_fingers": {
                "mjcf": relpath(full_mjcf_path, workspace_root),
                "urdf": relpath(full_urdf_path, workspace_root),
                "metadata": relpath(full_metadata_path, workspace_root),
                "body_count": full_metadata["counts"]["body_count"],
                "articulated_joint_count": full_metadata["counts"]["articulated_joint_count"],
            },
            "hands_only": {
                "metadata": relpath(hands_metadata_path, workspace_root),
                "assets": {
                    hand_key: {
                        "mjcf": metadata["paths"]["mjcf"],
                        "urdf": metadata["paths"]["urdf"],
                        "body_count": metadata["counts"]["body_count"],
                        "articulated_joint_count": metadata["counts"]["articulated_joint_count"],
                    }
                    for hand_key, metadata in standalone_full_hands.items()
                },
            },
            "palms_only": {
                "metadata": relpath(palms_metadata_path, workspace_root),
                "assets": {
                    hand_key: {
                        "mjcf": metadata["paths"]["mjcf"],
                        "urdf": metadata["paths"]["urdf"],
                        "body_count": metadata["counts"]["body_count"],
                        "articulated_joint_count": metadata["counts"]["articulated_joint_count"],
                    }
                    for hand_key, metadata in standalone_palms.items()
                },
            },
        },
    }
    overview_path.write_text(json.dumps(overview, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
