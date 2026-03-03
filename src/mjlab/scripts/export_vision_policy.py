"""Export a trained mjlab G1 vision policy checkpoint to ONNX.

Usage (from mjlab root):
    uv run export_vision_policy \\
        --checkpoint logs/rsl_rl/g1_velocity_vision/<run-id>/model_30000.pt \\
        --out-dir    exports/velocity_vision

The exported policy.onnx is then dropped into:
    <unitree_rl_mjlab>/deploy/robots/g1/config/policy/velocity_vision/v0/exported/

The observation normaliser (running mean/var) is baked into the ONNX graph so
the C++ deployment does NOT need to normalise separately.

If VelocityOnPolicyRunner was used during training, ONNX files are already
auto-exported next to every .pt checkpoint — check for a .onnx file in the
same folder first before running this script.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro


@dataclass(frozen=True)
class ExportConfig:
    checkpoint: Path
    """Path to the .pt checkpoint (e.g. logs/rsl_rl/g1_velocity_vision/.../model_30000.pt)."""

    out_dir: Path = Path("exports/velocity_vision")
    """Directory to write policy.onnx into."""

    task: str = "Mjlab-Velocity-Vision-Rough-Unitree-G1"
    """Registered mjlab task name — used to reconstruct env + runner for metadata."""

    device: str = "cpu"
    """Device for export ('cpu' is fine; no GPU needed for ONNX export)."""

    num_envs: int = 1
    """Number of parallel envs to instantiate (1 is enough for export)."""

    verbose: bool = False
    """Print ONNX graph details during export."""


def main(cfg: ExportConfig) -> None:
    checkpoint = cfg.checkpoint.resolve()
    if not checkpoint.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint}")
        sys.exit(1)

    print(f"[INFO] Loading checkpoint: {checkpoint}")

    # ------------------------------------------------------------------ #
    # Import mjlab tasks to populate the task registry.                    #
    # ------------------------------------------------------------------ #
    import mjlab.tasks  # noqa: F401  registers all tasks

    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper, MjlabOnPolicyRunner
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()

    # ------------------------------------------------------------------ #
    # Build env + runner (no training/simulation, just to reconstruct the  #
    # policy architecture and metadata).                                   #
    # ------------------------------------------------------------------ #
    env_cfg = load_env_cfg(cfg.task, play=True)
    agent_cfg = load_rl_cfg(cfg.task)

    env_cfg.scene.num_envs = cfg.num_envs

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
    vec_env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(cfg.task) or MjlabOnPolicyRunner
    runner = runner_cls(vec_env, asdict(agent_cfg), log_dir=None, device=cfg.device)
    runner.load(str(checkpoint), map_location=cfg.device)

    # ------------------------------------------------------------------ #
    # Export ONNX + attach metadata.                                       #
    # ------------------------------------------------------------------ #
    out_dir = str(cfg.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    filename = "policy.onnx"

    runner.export_policy_to_onnx(out_dir, filename=filename, verbose=cfg.verbose)

    from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
    run_name = f"exported_from_{checkpoint.stem}"
    metadata = get_base_metadata(raw_env, run_name)
    onnx_path = os.path.join(out_dir, filename)
    attach_metadata_to_onnx(onnx_path, metadata)

    print(f"\n[OK] ONNX exported → {onnx_path}")
    print(f"     Obs terms : {metadata['observation_names']}")
    print(f"     Joints    : {len(metadata['joint_names'])} DOF")
    print(
        f"\nCopy to deployment:\n"
        f"  cp {onnx_path} \\\n"
        f"    <unitree_rl_mjlab>/deploy/robots/g1/"
        f"config/policy/velocity_vision/v0/exported/policy.onnx"
    )

    raw_env.close()


def entry_point() -> None:
    main(tyro.cli(ExportConfig))


if __name__ == "__main__":
    entry_point()
