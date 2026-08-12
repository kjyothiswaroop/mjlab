import os
import sys

# Default to EGL for GPU-accelerated offscreen rendering on Linux. Must be set
# before any mujoco import: mujoco's gl_context module captures MUJOCO_GL once
# at load time. Override with e.g. MUJOCO_GL=osmesa on clusters without EGL.
# Linux-only because mujoco's gl_context rejects "egl" on macOS/Windows and
# raises at import. On those platforms we leave MUJOCO_GL alone so mujoco
# defaults to GLFW.
if sys.platform.startswith("linux"):
  os.environ.setdefault("MUJOCO_GL", "egl")

import ctypes
import glob
import platform
import sysconfig
import traceback
from importlib.metadata import entry_points
from pathlib import Path

import tyro
import warp as wp

MJLAB_SRC_PATH: Path = Path(__file__).parent

TYRO_FLAGS = (
  # Don't let users switch between types in unions. This produces a simpler CLI
  # with flatter helptext, at the cost of some flexibility. Type changes can
  # just be done in code.
  tyro.conf.AvoidSubcommands,
  # Disable automatic flag conversion (e.g., use `--flag False` instead of
  # `--no-flag` for booleans).
  tyro.conf.FlagConversionOff,
  # Use Python syntax for collections: --tuple (1,2,3) instead of --tuple 1 2 3.
  # Helps with wandb sweep compatibility: https://brentyi.github.io/tyro/wandb_sweeps/
  tyro.conf.UsePythonSyntaxForLiteralCollections,
)


def _preload_jetson_native_libs() -> None:
  """Preload NVPL/cuDSS libs needed by the Jetson (aarch64) CUDA torch wheel.

  That wheel dynamically links against NVPL (Arm Performance Libraries) and
  cuDSS but, unlike its other CUDA libs, doesn't declare them as dependencies
  torch can preload itself, so `import torch` fails with a bare
  `libnvpl_lapack_lp64_gomp.so.0: cannot open shared object file` unless
  these are loaded first. Setting LD_LIBRARY_PATH doesn't help here since
  it's only consulted by the dynamic linker at process start, not for
  ctypes/dlopen calls issued after Python is already running, so we preload
  the .so files directly by absolute path instead. Must run before anything
  imports torch, i.e. before `_import_registered_packages`.
  """
  if not (sys.platform.startswith("linux") and platform.machine() == "aarch64"):
    return

  site_packages = Path(sysconfig.get_paths()["purelib"])
  for name in ("libnvpl_blas_lp64_gomp.so.0", "libnvpl_lapack_lp64_gomp.so.0"):
    path = site_packages / "nvpl" / "lib" / name
    if path.exists():
      try:
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
      except OSError:
        pass

  cudss_glob = str(site_packages / "nvidia" / "cu*" / "lib" / "libcudss.so*")
  for lib_path in sorted(glob.glob(cudss_glob)):
    try:
      ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
    except OSError:
      pass


def _configure_warp() -> None:
  """Configure Warp globally for mjlab."""
  wp.config.enable_backward = False

  # Keep warp verbose by default to show kernel compilation progress.
  # Override with MJLAB_WARP_QUIET=1 environment variable if needed.
  quiet = os.environ.get("MJLAB_WARP_QUIET", "0").lower() in ("1", "true", "yes")
  wp.config.quiet = quiet


def _import_registered_packages() -> None:
  """Auto-discover and import packages registered via entry points.

  Looks for packages registered under the 'mjlab.tasks' entry point group.
  Each discovered package is imported, which allows it to register custom
  environments with gymnasium.
  """
  mjlab_tasks = entry_points().select(group="mjlab.tasks")
  for entry_point in mjlab_tasks:
    try:
      entry_point.load()
    except Exception:
      print(
        f"[WARN] Failed to load task package '{entry_point.name}' ({entry_point.value}):",
        file=sys.stderr,
      )
      traceback.print_exc(file=sys.stderr)


def _configure_mediapy() -> None:
  """Point mediapy at the bundled imageio-ffmpeg binary."""
  import imageio_ffmpeg
  import mediapy

  mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())


_preload_jetson_native_libs()
_configure_warp()
_configure_mediapy()
_import_registered_packages()
