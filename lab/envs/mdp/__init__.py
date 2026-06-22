"""MDP terms for the residual-RL teleop task.

Re-exports IsaacLab's stock terms (``isaaclab.envs.mdp``) plus this task's
custom action / observation / reward / event terms.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .actions import (  # noqa: F401
    RecordedQposResidualAction,
    RecordedQposResidualActionCfg,
)
from .events import (  # noqa: F401
    attach_demo,
    reset_object_to_demo_frame_0,
    reset_robot_to_demo_frame_0,
)
from .observations import (  # noqa: F401
    object_ang_vel_w,
    object_lin_vel_w,
    object_position_in_robot_root_frame,
    ref_base_action,
    ref_dof_delta,
    ref_object_pos_delta,
    ref_object_quat,
    ref_object_quat_delta,
)
from .rewards import (  # noqa: F401
    action_rate_l2,
    applied_torque_l2,
    hand_joint_vel_l2,
    inhand_object_stability,
    terminal_track_object_pos,
    terminal_track_object_rot,
    track_joint_pos,
    track_object_pos,
    track_object_rot,
    true_action_rate_l2,
)
