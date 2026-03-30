import argparse
import importlib
import time
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="Spawn an object on the pandaorca scene")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--fps", type=float, default=60.0, help="Simulation frame rate (default: 60)")
parser.add_argument(
	"--mode",
	type=str,
	default="dual",
	choices=["single", "dual"],
	help="Choose between single arm (right) or dual arm setup (default: dual)",
)
parser.add_argument(
	"--object",
	type=str,
	default="grape",
	choices=["ball", "duck", "fish", "grape", "shovel"],
	help="Object to spawn",
)
parser.add_argument(
	"--position",
	type=float,
	nargs=3,
	default=[0.0, 0.0, 1.0],
	metavar=("X", "Y", "Z"),
	help="Spawn position in meters (default: 0 0 1.0; table-top center)",
)
parser.add_argument(
	"--scale",
	type=float,
	default=1.0,
	help="Uniform object scale factor (default: 1.0)",
)
args = parser.parse_args()

simulation_app = SimulationApp(
	{
		"headless": args.headless,
		"renderer": "RayTracedLighting",
		"width": 1280,
		"height": 720,
	}
)

import numpy as np
import omni
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

SCRIPT_DIR = Path(__file__).parent
OBJECTS_DIR = SCRIPT_DIR / "../objects"
if args.mode == "single":
	USD_PATH = SCRIPT_DIR / "../../pandaorca_description/usd/pandaorca_single.usd"
elif args.mode == "dual":
	USD_PATH = SCRIPT_DIR / "../../pandaorca_description/usd/pandaorca_dual.usd"
else:
	raise ValueError(f"Invalid mode: {args.mode}")

# Prim paths in the stage
FRANKA_LEFT_PATH = "/World/fer_orcahand_left_extended"
FRANKA_RIGHT_PATH = "/World/fer_orcahand_right_extended"

# Number of DOFs expected from the h5 data
N_ARM_DOFS = 7
N_HAND_DOFS = 17

# From invkin_pose.py
ARM_JOINT_VALUES_INITIAL = np.array([0.184280, -0.225290, -0.193213, -2.701355, -0.069782, 2.478938, 0.050881])
HAND_JOINT_VALUES_INITIAL = np.array([0.0] * N_HAND_DOFS)


def setup_articulation(prim_path: str, world: World) -> SingleArticulation:
	name = prim_path.lstrip("/").replace("/", "_")
	art = SingleArticulation(prim_path=prim_path, name=name)
	world.scene.add(art)
	return art


def _enable_collision_and_gravity(root_prim: Usd.Prim) -> None:
	# Make object dynamic and affected by gravity.
	rb_api = UsdPhysics.RigidBodyAPI.Apply(root_prim)
	rb_api.CreateRigidBodyEnabledAttr(True)

	try:
		physx_rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
		physx_rb_api.CreateDisableGravityAttr(False)
	except Exception:
		pass

	for prim in Usd.PrimRange(root_prim):
		if prim.IsA(UsdGeom.Mesh):
			UsdPhysics.CollisionAPI.Apply(prim)
			mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
			mesh_collision_api.CreateApproximationAttr("convexHull")


def spawn_object(stage, object_name: str, position: list[float], scale: float) -> str:
	try:
		kit_commands = importlib.import_module("omni.kit.commands")
	except Exception as exc:
		raise RuntimeError("omni.kit.commands extension is required to spawn OBJ assets") from exc

	obj_path = (OBJECTS_DIR / object_name / f"{object_name}.obj").resolve()
	if not obj_path.exists():
		raise FileNotFoundError(f"Object mesh not found: {obj_path}")

	object_root_path = "/World/spawned_objects"
	UsdGeom.Xform.Define(stage, object_root_path)

	prim_path = f"{object_root_path}/{object_name}_instance"
	UsdGeom.Xform.Define(stage, prim_path)

	ref_path = f"{prim_path}/visual"
	kit_commands.execute(
		"CreateReferenceCommand",
		usd_context=omni.usd.get_context(),
		path_to=ref_path,
		asset_path=str(obj_path),
		instanceable=False,
	)

	prim = stage.GetPrimAtPath(prim_path)
	if not prim or not prim.IsValid():
		raise RuntimeError(f"Failed to create object prim at {prim_path}")

	xform_api = UsdGeom.XformCommonAPI(prim)
	xform_api.SetTranslate(Gf.Vec3d(position[0], position[1], position[2]))
	xform_api.SetScale(Gf.Vec3f(scale, scale, scale))
	_enable_collision_and_gravity(prim)
	return prim_path


def main():
	try:
		# Load USD scene
		omni.usd.get_context().open_stage(str(USD_PATH))

		# Create world and add articulations
		world = World()
		franka_right = setup_articulation(FRANKA_RIGHT_PATH, world)
		if args.mode == "dual":
			franka_left = setup_articulation(FRANKA_LEFT_PATH, world)
		world.reset()

		# Set initial joint values for arm and hand
		q0 = np.concatenate([ARM_JOINT_VALUES_INITIAL, HAND_JOINT_VALUES_INITIAL])
		franka_right.set_joint_positions(q0)
		if args.mode == "dual":
			franka_left.set_joint_positions(q0)

		# Spawn selected object at default table center (0, 0, 1.0) unless overridden.
		stage = omni.usd.get_context().get_stage()
		spawned_prim_path = spawn_object(stage, args.object, args.position, args.scale)
		print(f"Spawned '{args.object}' at {args.position} with scale {args.scale} -> {spawned_prim_path}")

		# Simulation loop
		dt = 1.0 / max(args.fps, 1e-6)
		while simulation_app.is_running():
			world.step(render=True)
			time.sleep(dt)
	finally:
		simulation_app.close()


if __name__ == "__main__":
	main()
