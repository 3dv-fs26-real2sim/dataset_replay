"""Robot articulation setup, DOF resolution, and collision control.

Imports Isaac Sim types — must be imported after SimulationApp is created.
"""

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation


def setup_articulation(prim_path: str, world: World) -> SingleArticulation:
    """Create a robot articulation from a USD prim path and add it to the world."""
    name = prim_path.lstrip("/").replace("/", "_")
    art = SingleArticulation(prim_path=prim_path, name=name)
    world.scene.add(art)
    return art


def print_dof_info(label: str, art: SingleArticulation) -> None:
    """Print DOF names and indices for debugging."""
    print(f"\n[DOF] {label}: {art.num_dof} DOFs")
    for i, name in enumerate(art.dof_names):
        print(f"      [{i:2d}] {name}")


def resolve_dof_indices(
    art: SingleArticulation, names: list[str], label: str
) -> np.ndarray:
    """Map canonical joint names to articulation DOF indices.

    Supports alias matching (panda_joint ↔ fer_joint) and suffix matching as fallback.
    """
    dof_names = list(art.dof_names)
    name_to_idx = {n: i for i, n in enumerate(dof_names)}

    def candidate_names(name: str) -> list[str]:
        if name.startswith("panda_joint"):
            return [name, name.replace("panda_joint", "fer_joint", 1)]
        if name.startswith("fer_joint"):
            return [name, name.replace("fer_joint", "panda_joint", 1)]
        return [name]

    indices = []
    for name in names:
        resolved = False
        for cand in candidate_names(name):
            if cand in name_to_idx:
                if cand != name:
                    print(f"[DOF] '{name}' matched via alias '{cand}'")
                indices.append(name_to_idx[cand])
                resolved = True
                break
        if resolved:
            continue

        matches = [dof for dof in dof_names if dof.endswith(name)]
        if len(matches) == 1:
            print(f"[DOF] '{name}' matched via suffix to '{matches[0]}'")
            indices.append(name_to_idx[matches[0]])
        elif len(matches) > 1:
            raise RuntimeError(
                f"[DOF] Ambiguous suffix match for '{name}' in {label}: {matches}"
            )
        else:
            raise RuntimeError(
                f"[DOF] Cannot find '{name}' in {label} DOFs: {dof_names}"
            )

    return np.array(indices, dtype=int)


def set_collision_enabled(stage, prim_path: str, enabled: bool) -> None:
    """Enable or disable collision on a prim."""
    from pxr import UsdPhysics

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"[collision] WARNING: {prim_path} not found in stage")
        return

    collision_api = UsdPhysics.CollisionAPI(prim)
    collision_api.GetCollisionEnabledAttr().Set(enabled)
    state = "Enabled" if enabled else "Disabled"
    print(f"[collision] {state} collision on {prim_path}")
