"""Narrow, explicit URDF visual/kinematic import, with an independent URDF FK evaluator.

Supports fixed/revolute/prismatic tree joints, local mesh files and box visuals.
Collisions/inertias are retained upstream, never substituted for visual geometry.
"""
from __future__ import annotations

import hashlib
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from turnitover.assets.catalog import validate_spec
from turnitover.core.spec import AssetSpec, JointSpec, JointType, MaterialSpec, MeshSpec, PartSpec


def vector(text, default=(0, 0, 0)):
    result = np.asarray(default if text is None else [float(x) for x in text.replace(",", " ").split()], dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("URDF vector must contain three finite numbers")
    return result


def origin(element):
    node = element.find("origin")
    result = np.eye(4)
    if node is not None:
        result[:3, :3] = Rotation.from_euler("xyz", vector(node.get("rpy"))).as_matrix()
        result[:3, 3] = vector(node.get("xyz"))
    return result


def xyz_angles(matrix):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # gimbal lock has a valid equivalent Euler representation
        return tuple(float(x) for x in Rotation.from_matrix(matrix[:3, :3]).as_euler("XYZ"))


class UrdfModel:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.root = ET.parse(path).getroot()
        if self.root.tag != "robot":
            raise ValueError("Expected a URDF robot")
        self.links = {e.attrib["name"]: e for e in self.root.findall("link")}
        self.joints = {e.attrib["name"]: e for e in self.root.findall("joint")}
        if len(self.links) != len(self.root.findall("link")) or len(self.joints) != len(self.root.findall("joint")):
            raise ValueError("Duplicate URDF link or joint")
        self.by_child = {}
        for name, joint in self.joints.items():
            if joint.get("type") not in ("fixed", "revolute", "prismatic") or joint.find("mimic") is not None:
                raise ValueError(f"Unsupported joint type or mimic: {name}")
            parent, child = joint.find("parent").attrib["link"], joint.find("child").attrib["link"]
            if child in self.by_child or child not in self.links or parent not in self.links:
                raise ValueError("URDF must be a tree of known links")
            self.by_child[child] = joint
            if joint.get("type") != "fixed":
                lo, hi = self.limits(joint)
                if not lo <= 0 <= hi or lo == hi or not np.isfinite([lo, hi]).all():
                    raise ValueError("Finite nonzero joint travel including rest=0 is required")
                self.axis(joint)
        roots = set(self.links) - set(self.by_child)
        if len(roots) != 1:
            raise ValueError("URDF must have exactly one root")
        self.root_link = next(iter(roots))
        self.transforms({})  # rejects cycles/disconnected components
        self.files = {self.path}
        self.meshes = {name: self._visual_meshes(link) for name, link in self.links.items()}

    @staticmethod
    def limits(joint):
        node = joint.find("limit")
        if node is None:
            raise ValueError("Movable joints require limits")
        return float(node.attrib["lower"]), float(node.attrib["upper"])

    @staticmethod
    def axis(joint):
        node = joint.find("axis")
        axis = vector(node.get("xyz") if node is not None else None, (1, 0, 0))
        if np.linalg.norm(axis) == 0:
            raise ValueError("Zero joint axis")
        return axis / np.linalg.norm(axis)

    def transforms(self, values):
        if set(values) - set(self.joints):
            raise ValueError("Unknown joint value")
        result, visiting = {}, set()
        def visit(link):
            if link in visiting:
                raise ValueError("Cycle in URDF tree")
            if link in result:
                return result[link]
            visiting.add(link)
            joint = self.by_child.get(link)
            if joint is None:
                pose = np.eye(4)
            else:
                pose = visit(joint.find("parent").attrib["link"]) @ origin(joint)
                if joint.get("type") != "fixed":
                    value = float(values.get(joint.attrib["name"], 0))
                    lo, hi = self.limits(joint)
                    if not np.isfinite(value) or not lo <= value <= hi:
                        raise ValueError("Joint value out of range")
                    motion = np.eye(4)
                    if joint.get("type") == "revolute":
                        motion[:3, :3] = Rotation.from_rotvec(self.axis(joint) * value).as_matrix()
                    else:
                        motion[:3, 3] = self.axis(joint) * value
                    pose = pose @ motion
            visiting.remove(link)
            result[link] = pose
            return pose
        for link in self.links:
            visit(link)
        return result

    def _visual_meshes(self, link):
        meshes = []
        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            if geometry is None or len(geometry) != 1:
                raise ValueError("Each visual must define exactly one geometry")
            shape = geometry[0]
            if shape.tag == "mesh":
                filename = shape.attrib["filename"]
                if "://" in filename:
                    raise ValueError("Only resolved local URDF mesh paths are supported")
                path = (self.path.parent / filename).resolve()
                if not path.is_file():
                    raise ValueError(f"Missing mesh: {path}")
                self.files.add(path)
                scene = trimesh.load(path, force="scene", process=False)
                scale = vector(shape.get("scale"), (1, 1, 1))
                if np.any(scale <= 0):
                    raise ValueError("Positive mesh scale is required")
                for node in sorted(scene.graph.nodes_geometry):
                    transform, key = scene.graph[node]
                    mesh = scene.geometry[key].copy()
                    mesh.apply_transform(origin(visual) @ np.diag([*scale, 1]) @ transform)
                    meshes.append(mesh)
            elif shape.tag == "box":
                size = vector(shape.get("size"))
                if np.any(size <= 0):
                    raise ValueError("Positive box dimensions required")
                mesh = trimesh.creation.box(extents=size)
                mesh.apply_transform(origin(visual))
                meshes.append(mesh)
            else:
                raise ValueError(f"Unsupported visual geometry: {shape.tag}")
        return meshes

    def spec(self, asset_id, category):
        parts, joints = [], []
        for name in self.links:
            joint = self.by_child.get(name)
            frame = origin(joint) if joint is not None else np.eye(4)
            parent = joint.find("parent").attrib["link"] if joint is not None else None
            active = joint is not None and joint.get("type") != "fixed"
            vertices, faces, colors = [], [], []
            for mesh in self.meshes[name]:
                offset = len(vertices)
                vertices.extend(tuple(float(x) for x in v) for v in mesh.vertices)
                faces.extend(tuple(int(x) + offset for x in f) for f in mesh.faces)
                visual = mesh.visual.to_color() if mesh.visual.kind == "texture" else mesh.visual
                rgb = np.asarray(visual.vertex_colors)[:, :3] / 255.
                rgb = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
                colors.extend(tuple(float(x) for x in c) for c in rgb)
            parts.append(PartSpec(name, parent, tuple(float(x) for x in frame[:3, 3]),
                                  rotation=(0., 0., 0.) if active else xyz_angles(frame),
                                  mesh=MeshSpec(tuple(vertices), tuple(faces), tuple(colors))))
            if active:
                joints.append(JointSpec(joint.attrib["name"], name, JointType(joint.attrib["type"]),
                    tuple(float(x) for x in self.axis(joint)), tuple(float(x) for x in frame[:3, 3]),
                    self.limits(joint), frame_rotation=xyz_angles(frame)))
        spec = AssetSpec(asset_id, category, tuple(parts), tuple(joints), (MaterialSpec("default"),))
        validate_spec(spec)
        return spec

    def world_vertices(self, values):
        transforms = self.transforms(values)
        return {name: trimesh.transform_points(np.concatenate([m.vertices for m in meshes]), transforms[name])
                for name, meshes in self.meshes.items() if meshes}

    def provenance(self):
        return [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(self.files)]
