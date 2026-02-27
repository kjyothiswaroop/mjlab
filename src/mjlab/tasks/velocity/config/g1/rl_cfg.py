"""RL configuration for Unitree G1 velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_g1_vision_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """RL runner config for G1 vision policy with CNN depth encoder.

  The actor uses CNNModel: proprioceptive obs ("actor" group, 1D) go through
  an MLP stream while depth image ("actor_depth" group, shape B×1×H×W) goes
  through a CNN encoder. Both latents are concatenated before the shared MLP
  head — matching the legged-loco DepthOnlyFCBackbone dual-stream approach.

  CNN architecture for 30×53 depth input:
    Conv(1→16, k=5) → MaxPool  →  (25, 13)
    Conv(16→32, k=3) → MaxPool →  (12, 6)
    Flatten → 2304 dims
  Combined latent (proprio MLP + CNN): fed into hidden_dims MLP → actions.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      stochastic=True,
      init_noise_std=1.0,
      class_name="CNNModel",
      cnn_cfg={
        "output_channels": [16, 32],
        "kernel_size": [5, 3],
        "max_pool": [True, True],
        "activation": "elu",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      stochastic=False,
      init_noise_std=1.0,
    ),
    obs_groups={"actor": ("actor", "actor_depth"), "critic": ("critic",)},
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_velocity_vision",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


def unitree_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      stochastic=True,
      init_noise_std=1.0,
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      stochastic=False,
      init_noise_std=1.0,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_velocity",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=30_000,
  )
