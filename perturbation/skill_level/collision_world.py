"""Construct mplib FCLObjects for the workspace collision world.

What we add:
  1. Table plane below the gripper (z<=0) — large flat box at z = -0.05 thickness 0.1
     so the EE can never plan a path that dips below 0 without flagging collision.
  2. (optional) Cylindrical workspace ceiling — a "no-fly above" cap so plans
     don't take wild detours up into the ceiling. Off by default.
  3. (optional) Other-arm exclusion box — for bi-arm setups where the second
     SO-101 robot is positioned at known offset; treats it as a single AABB.

What we deliberately do NOT add (yet):
  - detected scene objects (blocks, dishes). These belong in a future "live
    scene" update path that pushes detected positions into the collision world
    each subtask. For now, the gripper just plans around static workspace.
  - reach cylinder. The arm's reach limit is already enforced by URDF joint
    limits + mplib's IK; an extra cylinder would over-constrain.

Usage::

    from perturbation.skill_level.collision_world import build_workspace_objects

    objects = build_workspace_objects(WorkspaceConfig(
        table_z=0.0, table_thickness=0.1, table_size=(2.0, 2.0),
    ))
    planner = mplib.Planner(urdf=..., move_group=..., objects=objects)

These objects show up in ``planner.planning_world.check_collision()`` and are
honored by every OMPL state-validity check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class WorkspaceConfig:
    """Workspace collision-world parameters.

    Coordinates are in the robot's base_link frame: +x forward, +y left,
    +z up. Table surface is at z=0 by default.
    """
    # Table — extends infinitely in xy, finite thickness, top at z=table_z
    table_enabled: bool = True
    table_z: float = 0.0
    table_thickness: float = 0.1            # m — keep thick to forbid plans dipping below
    table_size: tuple[float, float] = (2.0, 2.0)  # x, y extents (m)

    # Optional ceiling — flat box above the workspace
    ceiling_enabled: bool = False
    ceiling_z: float = 0.40
    ceiling_thickness: float = 0.05

    # Optional other-arm AABB (bi-arm setups). One AABB; pose+size in robot frame.
    other_arms: Sequence[dict] = field(default_factory=list)
    # each dict: {"name": str, "size": (x,y,z), "center": (x,y,z)}

    # Robot links that are allowed to touch the workspace_table (e.g., base_link
    # is mounted ON the table — without this whitelist every state collides).
    table_mount_links: tuple[str, ...] = ("base_link",)


def build_workspace_objects(cfg: WorkspaceConfig) -> list:
    """Return a list of FCLObjects ready to pass into mplib.Planner(objects=...)."""
    from mplib.collision_detection.fcl import (
        Box, CollisionObject, FCLObject,
    )
    from mplib.pymp import Pose

    objects = []

    if cfg.table_enabled:
        sx, sy = cfg.table_size
        thick = cfg.table_thickness
        # Box centered at z = table_z - thick/2 so its TOP face is at table_z.
        box_geom = Box(sx, sy, thick)
        center_z = cfg.table_z - thick / 2.0
        co = CollisionObject(box_geom, Pose([0.0, 0.0, center_z], [1.0, 0.0, 0.0, 0.0]))
        fcl_obj = FCLObject(
            "workspace_table",
            Pose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            [co],
            [Pose([0.0, 0.0, center_z], [1.0, 0.0, 0.0, 0.0])],
        )
        objects.append(fcl_obj)

    if cfg.ceiling_enabled:
        sx, sy = cfg.table_size  # reuse footprint
        thick = cfg.ceiling_thickness
        box_geom = Box(sx, sy, thick)
        center_z = cfg.ceiling_z + thick / 2.0
        co = CollisionObject(box_geom, Pose([0.0, 0.0, center_z], [1.0, 0.0, 0.0, 0.0]))
        fcl_obj = FCLObject(
            "workspace_ceiling",
            Pose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            [co],
            [Pose([0.0, 0.0, center_z], [1.0, 0.0, 0.0, 0.0])],
        )
        objects.append(fcl_obj)

    for i, arm in enumerate(cfg.other_arms):
        name = arm.get("name", f"other_arm_{i}")
        size = arm["size"]      # (x, y, z) in m
        center = arm["center"]  # (x, y, z) in m
        box_geom = Box(*size)
        co = CollisionObject(box_geom, Pose(list(center), [1.0, 0.0, 0.0, 0.0]))
        fcl_obj = FCLObject(
            name,
            Pose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            [co],
            [Pose(list(center), [1.0, 0.0, 0.0, 0.0])],
        )
        objects.append(fcl_obj)

    return objects
