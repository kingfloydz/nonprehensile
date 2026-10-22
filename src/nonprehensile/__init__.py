"""GPU tray manipulation and mode-dependent acceleration rewards."""

import os

# Training uses state observations, so avoid loading OpenGL on headless workers.
# Set this before any submodule imports MuJoCo; preserve explicit render settings.
os.environ.setdefault("MUJOCO_GL", "disable")
