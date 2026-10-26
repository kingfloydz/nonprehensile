"""Shared physical settings for the four-mode tray-object benchmark.

Dimensions are full lengths in meters. Actions are [alpha_x,y,z, a_x,y,z].
Every policy controls all six components, including vertical acceleration.
"""

from dataclasses import dataclass
from math import radians

BENCHMARK_VERSION = "four-mode-v2"
TASKS = ("sticking", "sliding", "tipping", "twisting")


@dataclass(frozen=True)
class TaskSettings:
    goal: float
    size_low: tuple
    size_high: tuple
    friction: tuple
    action_limits: tuple
    speed_limit: float
    angular_speed_limit: float
    initial_tilt_deg: float = 0.0
    initial_xy_offset: float = 0.0


SETTINGS = {
    "sticking": TaskSettings(
        0.5,
        (0.15, 0.15, 0.10),
        (0.25, 0.25, 0.20),
        (0.20, 0.50),
        (2.0, 2.0, 2.0, 2.0, 1.0, 1.0),
        1.5,
        2.0,
    ),
    "sliding": TaskSettings(
        0.2,
        (0.15, 0.10, 0.05),
        (0.25, 0.20, 0.08),
        (0.15, 0.30),
        (2.0, 2.0, 2.0, 5.0, 1.0, 1.0),
        3.0,
        2.0,
        initial_tilt_deg=2.0,
    ),
    "tipping": TaskSettings(
        radians(80),
        (0.10, 0.10, 0.25),
        (0.12, 0.15, 0.30),
        (0.60, 0.80),
        (2.0, 2.0, 2.0, 6.0, 1.0, 1.0),
        3.0,
        2.0,
    ),
    "twisting": TaskSettings(
        radians(45),
        (0.25, 0.25, 0.05),
        (0.30, 0.30, 0.08),
        (0.05, 0.12),
        (2.0, 2.0, 30.0, 2.0, 2.0, 2.0),
        1.5,
        8.0,
        initial_xy_offset=0.005,
    ),
}

# Fixed across methods and action-bound changes: these define perturbation units.
METRIC_SCALE = (30.0, 30.0, 30.0, 15.0, 15.0, 15.0)
