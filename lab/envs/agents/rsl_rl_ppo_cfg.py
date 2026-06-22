"""rsl_rl PPO runner config for the residual-RL teleop task.

Targets the ``rsl-rl-lib`` 3.x schema (IsaacLab 2.3.x): the runner needs an
``obs_groups`` mapping from the algorithm's actor/critic sets to the env's
observation groups, and observation normalization is set per-network on the
policy cfg (``actor_obs_normalization`` / ``critic_obs_normalization``) rather
than the deprecated ``empirical_normalization`` flag.
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class TeleopPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """On-policy PPO runner for the Franka + OrcaHand residual policy."""

    num_steps_per_env = 32
    max_iterations = 10000
    save_interval = 100
    experiment_name = "teleop_residual"
    clip_actions = 1.0

    # Map the algorithm's actor/critic obs sets to the env's obs groups. The env
    # exposes asymmetric "policy" (actor) and "critic" (privileged) groups.
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.0,
        desired_kl=0.008,
        max_grad_norm=1.0,
        value_loss_coef=4.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        num_learning_epochs=5,
        num_mini_batches=4,
    )
