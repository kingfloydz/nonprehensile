from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg

from .geometry import acceleration_constraints, measures
from .math import mv, quat_step, rotation
from .model import XML
from .settings import BENCHMARK_VERSION, METRIC_SCALE, SETTINGS


@dataclass
class EnvConfig:
    task: str = "sticking"
    group: str = "A"
    num_envs: int = 4096
    seed: int = 1
    distance_weight: float = 1.0
    margin_weight: float = 0.1
    conventional_weight: float = 1.0
    startup_weight: float = 1.0
    margin_reference: float = 0.05
    margin_temperature: float = 0.05
    projection_iterations: int = 48
    mapping_iterations: int = 32
    disturbance: float = 0.0
    discount: float = 0.99**0.25
    translation_tolerance: float = 0.001
    rotation_tolerance: float = float(np.deg2rad(1))
    gap_tolerance: float = 0.0005
    min_normal_load: float = 0.1  # 20 ms average, in units of mg
    tray_speed_limit: float | None = None
    tray_angular_speed_limit: float | None = None
    action_limits: tuple | None = None
    goal: float | None = None
    control_dt: float = 0.005
    physics_dt: float = 0.00025
    timeout_s: float = 5.0
    benchmark_version: str = BENCHMARK_VERSION

    def __post_init__(self):
        if self.benchmark_version != BENCHMARK_VERSION:
            raise ValueError("Benchmark version mismatch; rebuild banks and retrain.")
        spec = SETTINGS[self.task]
        for name, value in (
            ("goal", spec.goal),
            ("action_limits", spec.action_limits),
            ("tray_speed_limit", spec.speed_limit),
            ("tray_angular_speed_limit", spec.angular_speed_limit),
        ):
            if getattr(self, name) is None:
                setattr(self, name, value)
        if self.physics_dt <= 0 or self.control_dt < self.physics_dt:
            raise ValueError("Require 0 < physics_dt <= control_dt")
        if not np.isclose(
            self.control_dt / self.physics_dt, round(self.control_dt / self.physics_dt)
        ):
            raise ValueError("control_dt must be a multiple of physics_dt")
        if len(self.action_limits) != 6 or min(self.action_limits) <= 0:
            raise ValueError("Six positive acceleration limits are required")
        if self.timeout_s <= 0 or self.goal <= 0:
            raise ValueError("Timeout and goal must be positive")


class TrayEnv:
    """MuJoCo-Warp simulation in the accelerating tray frame.

    A fixed tray and an independently simulated payload avoid a servo or an
    artificially heavy tray. Exact non-inertial force/torque terms implement
    commanded tray acceleration. No artificial contact or mode enforcement.
    """

    @property
    def dt(self):
        return self.cfg.control_dt

    @property
    def substeps(self):
        return round(self.dt / self.cfg.physics_dt)

    @property
    def max_steps(self):
        return int(np.ceil(self.cfg.timeout_s / self.dt))

    def __init__(self, cfg, bank, device="cuda:0"):
        self.cfg, self.device, self.n = cfg, device, cfg.num_envs
        self.rng = torch.Generator(device=device).manual_seed(cfg.seed)
        self.noise_rng = torch.Generator(device=device).manual_seed(cfg.seed + 1000)
        with np.load(bank) as data:
            if (
                "benchmark_version" not in data
                or str(data["benchmark_version"]) != cfg.benchmark_version
            ):
                raise ValueError("Old or incompatible bank; rebuild with the current settings")
            self.bank = {
                k: torch.tensor(data[f"{cfg.task}_{k}"], device=device)
                for k in ("size", "mu", "h", "e", "qpos", "quaternion")
            }
        model = mujoco.MjModel.from_xml_string(XML)
        self.sim = Simulation(
            self.n,
            SimulationCfg(
                nconmax=16,
                njmax=64,
                mujoco=MujocoCfg(
                    timestep=cfg.physics_dt,
                    gravity=(0.0, 0.0, 0.0),
                    cone="elliptic",
                    iterations=30,
                    ls_iterations=10,
                ),
            ),
            model=model,
            device=device,
        )
        self.sim.expand_model_fields(
            (
                "geom_size",
                "geom_friction",
                "geom_rbound",
                "geom_aabb",
                "body_mass",
                "body_inertia",
                "body_subtreemass",
                "body_invweight0",
                "dof_invweight0",
            )
        )
        self.size = self.zeros(3)
        self.mu, self.mass = self.zeros(), self.zeros()
        self.diagonal = self.zeros(3)
        self.bank_ids = torch.zeros(self.n, dtype=torch.long, device=device)
        self.position, self.velocity, self.omega = self.zeros(3), self.zeros(3), self.zeros(3)
        self.quaternion = self.zeros(4)
        self.origin = self.zeros(3)
        self.entered = torch.zeros(self.n, dtype=torch.bool, device=device)
        self.steps = torch.zeros(self.n, dtype=torch.long, device=device)
        self.scale = torch.tensor(cfg.action_limits, device=device)
        self.load_history = self.zeros(max(1, round(0.02 / cfg.physics_dt)))
        self.load_cursor = 0
        self.normal_fraction = self.zeros()
        self.last_gap = self.zeros()
        self.last_no_contact = torch.zeros(self.n, dtype=torch.bool, device=device)
        self.all_ids = torch.arange(self.n, device=device)
        self.reset(self.all_ids)

    def zeros(self, *shape):
        return torch.zeros((self.n, *shape), device=self.device)

    def reset(self, ids, bank_indices=None):
        count = len(ids)
        indices = bank_indices
        if indices is None:
            indices = torch.randint(
                len(self.bank["size"]), (count,), device=self.device, generator=self.rng
            )
        self.bank_ids[ids] = indices
        size = self.bank["size"][indices]
        mass = torch.ones(count, device=self.device)
        inertia = mass[:, None] / 12 * (size.square().sum(-1, keepdim=True) - size.square())
        self.size[ids], self.mass[ids], self.diagonal[ids] = size, mass, inertia
        self.mu[ids] = self.bank["mu"][indices]
        self.sim.model.geom_size[ids, 1] = size / 2
        self.sim.model.geom_aabb[ids, 1, 1] = size / 2
        self.sim.model.geom_rbound[ids, 1] = (size / 2).norm(dim=-1)
        self.sim.model.geom_friction[ids, 1, 0] = self.mu[ids]
        self.sim.model.body_mass[ids, 1] = mass
        self.sim.model.body_inertia[ids, 1] = inertia
        self.sim.reset(ids)
        qpos = self.sim.data.qpos
        qpos[ids] = self.bank["qpos"][indices]
        self.origin[ids] = qpos[ids, :3]
        self.origin[ids, 2] = 0
        self.position[ids], self.velocity[ids], self.omega[ids] = 0, 0, 0
        self.quaternion[ids] = self.bank["quaternion"][indices]
        initial_load = rotation(self.quaternion[ids])[:, 2, 2]
        self.load_history[ids] = initial_load[:, None]
        self.normal_fraction[ids] = initial_load
        self.last_gap[ids], self.last_no_contact[ids] = 0, False
        self.steps[ids], self.entered[ids] = 0, False
        self.sim.recompute_constants(RecomputeLevel.set_const)
        self.sim.forward()

    def state(self):
        q, v = self.sim.data.qpos, self.sim.data.qvel
        rot = rotation(q[:, 3:7])
        gravity = mv(
            rotation(self.quaternion).transpose(1, 2),
            q.new_tensor([0.0, 0, -9.81]).expand(self.n, 3),
        )
        inertia = rot @ torch.diag_embed(self.diagonal) @ rot.transpose(1, 2)
        return q[:, :3], rot, v[:, :3], mv(rot, v[:, 3:6]), gravity, inertia

    def observe(self):
        c, r, vr, wr, g, _ = self.state()
        rt = rotation(self.quaternion).transpose(1, 2)
        goal = self.zeros(3)
        target = self.cfg.goal
        if self.cfg.task == "sticking":
            goal = mv(rt, c.new_tensor([target, 0, 0]) - self.position) / target
        elif self.cfg.task == "sliding":
            goal[:, 0] = (self.origin[:, 0] + target - c[:, 0]) / target
        else:
            angle = (
                torch.atan2(r[:, 0, 2], r[:, 0, 0])
                if self.cfg.task == "tipping"
                else torch.atan2(r[:, 1, 0], r[:, 0, 0])
            )
            goal[:, 0] = 1 - angle / target
        spec = SETTINGS[self.cfg.task]
        low, high = c.new_tensor(spec.size_low), c.new_tensor(spec.size_high)
        return torch.cat(
            (
                c / 0.4,
                r.flatten(1),
                vr / 2,
                wr / 3,
                mv(rt, self.velocity) / 2,
                self.omega / 3,
                g / 9.81,
                2 * (self.size - low) / (high - low) - 1,
                2 * (self.mu[:, None] - spec.friction[0]) / (spec.friction[1] - spec.friction[0])
                - 1,
                goal,
                self.entered[:, None].float(),
                self.steps[:, None] / self.max_steps,
            ),
            -1,
        )

    def constraints(self, startup=None):
        """Physical acceleration halfspaces/equalities; shared by rewards and F."""
        c, rot, vr, wr, gravity, inertia = self.state()
        if startup is None:
            startup = torch.zeros_like(self.entered)
        ids, task = self.bank_ids, self.cfg.task
        return acceleration_constraints(
            task,
            self.bank["h"][ids],
            self.bank["e"][ids],
            c,
            inertia,
            self.mass,
            self.omega,
            gravity,
            vr,
            wr,
            self.size,
            rot,
            self.origin,
            startup,
            dt=self.dt,
        )

    def physics(self, chi):
        alpha, accel = chi[:, :3], chi[:, 3:]
        # All modes latch their first event; later substeps cannot erase failures.
        event_success = torch.zeros_like(self.entered)
        event_failure = torch.zeros_like(self.entered)
        event_started = torch.zeros_like(self.entered)
        event_error = self.zeros()
        event_time = (self.steps + 1).float() * self.dt
        peak_error, peak_gap = self.zeros(), self.zeros()
        min_load = self.normal_fraction.clone()
        no_contact = torch.zeros_like(self.entered)
        peak_drop, event_velocity = self.zeros(), self.velocity.clone()
        event_speed, event_spin = self.zeros(), self.zeros()
        h = self.cfg.physics_dt
        for substep in range(self.substeps):
            c, rot, vr, wr, gravity, inertia = self.state()
            omega = self.omega
            total = omega + wr
            fictitious = gravity - accel - torch.linalg.cross(alpha, c)
            fictitious -= torch.linalg.cross(omega, torch.linalg.cross(omega, c))
            fictitious -= 2 * torch.linalg.cross(omega, vr)
            torque = -mv(inertia, alpha + torch.linalg.cross(omega, wr))
            torque -= torch.linalg.cross(total, mv(inertia, total))
            torque += torch.linalg.cross(wr, mv(inertia, wr))
            self.sim.data.qfrc_applied[:, :3] = self.mass[:, None] * fictitious
            self.sim.data.qfrc_applied[:, 3:] = mv(rot.transpose(1, 2), torque)
            world_accel = mv(rotation(self.quaternion), accel)
            self.position += self.velocity * h + world_accel * (0.5 * h * h)
            self.velocity += world_accel * h
            self.quaternion = quat_step(self.quaternion, omega + alpha * (0.5 * h), h)
            self.omega += alpha * h
            self.sim.step()
            # The only constraints on the free box are tray contacts. Translational
            # generalized constraint force is expressed in the tray coordinates.
            load = self.sim.data.qfrc_constraint[:, 2].clamp_min(0) / (self.mass * 9.81)
            self.load_history[:, self.load_cursor] = load
            self.load_cursor = (self.load_cursor + 1) % self.load_history.shape[1]
            self.normal_fraction = self.load_history.mean(1)
            success, failure, started, error = self.outcomes()
            active = ~(event_success | event_failure)
            first = active & (success | failure)
            peak_error = torch.where(active, torch.maximum(peak_error, error), peak_error)
            peak_gap = torch.where(active, torch.maximum(peak_gap, self.last_gap), peak_gap)
            min_load = torch.where(active, torch.minimum(min_load, self.normal_fraction), min_load)
            no_contact |= active & (load < 1e-6)
            peak_drop = torch.where(
                active, torch.maximum(peak_drop, -self.position[:, 2]), peak_drop
            )
            event_velocity = torch.where(active[:, None], self.velocity, event_velocity)
            event_speed = torch.where(
                active, torch.maximum(event_speed, self.velocity.norm(dim=1)), event_speed
            )
            event_spin = torch.where(
                active, torch.maximum(event_spin, self.omega.norm(dim=1)), event_spin
            )
            event_error = torch.where(active, error, event_error)
            event_started |= active & started
            event_success |= first & success & ~failure
            event_failure |= first & failure
            event_time = torch.where(
                first, self.steps.float() * self.dt + (substep + 1) * h, event_time
            )
        self.last_diagnostics = dict(
            mode_error=peak_error,
            contact_gap=peak_gap,
            min_normal_load=min_load,
            no_contact=no_contact,
            tray_drop=peak_drop,
            terminal_vertical_velocity=event_velocity[:, 2],
            max_tray_speed=event_speed,
            max_tray_angular_speed=event_spin,
        )
        return event_success, event_failure, event_started, event_error, event_time

    def outcomes(self):
        c, rot, vr, wr, _, _ = self.state()
        delta = c - self.origin
        task = self.cfg.task
        angle = torch.acos(((rot.diagonal(dim1=1, dim2=2).sum(1) - 1) / 2).clamp(-1, 1))
        tilt = torch.acos(rot[:, 2, 2].clamp(-1, 1))
        yaw = torch.atan2(rot[:, 1, 0], rot[:, 0, 0])
        pitch = torch.atan2(rot[:, 0, 2], rot[:, 0, 0])
        pos_tol, angle_tol = self.cfg.translation_tolerance, self.cfg.rotation_tolerance
        if task == "sticking":
            error = delta[:, :2].norm(dim=-1) / 0.003 + angle / 0.05236
            mode_fail = (delta[:, :2].norm(dim=-1) > pos_tol) | (angle > angle_tol)
            target = c.new_tensor([self.cfg.goal, 0.0, 0.0])
            success = (self.position - target).norm(dim=-1) < 0.005
            success &= (self.velocity.norm(dim=-1) < 0.03) & (self.omega.norm(dim=-1) < 0.1)
            started = torch.ones_like(success)
        elif task == "sliding":
            error = delta[:, 1].abs() / 0.003 + angle / 0.05236
            mode_fail = (delta[:, 1].abs() > pos_tol) | (angle > angle_tol) | (vr[:, 0] < -0.03)
            success = delta[:, 0] >= self.cfg.goal
            started = vr[:, 0] > 0.02
        elif task == "twisting":
            error = delta[:, :2].norm(dim=-1) / 0.003 + tilt / 0.05236
            mode_fail = (
                (delta[:, :2].norm(dim=-1) > pos_tol) | (tilt > angle_tol) | (wr[:, 2] < -0.1)
            )
            success = yaw >= self.cfg.goal
            started = wr[:, 2] > 0.1
        else:
            local = self.zeros(3)
            local[:, 0], local[:, 2] = self.size[:, 0] / 2, -self.size[:, 2] / 2
            edge = c + mv(rot, local)
            expected = self.origin.clone()
            expected[:, 0] += self.size[:, 0] / 2
            axis_offset = (rot[:, :, 1] - c.new_tensor([0.0, 1.0, 0.0])) * self.size[:, 1:2] / 2
            slip = torch.maximum(
                (edge - expected + axis_offset).norm(dim=-1),
                (edge - expected - axis_offset).norm(dim=-1),
            )
            axis_error = torch.acos(rot[:, 1, 1].clamp(-1, 1))
            error = slip / 0.003 + axis_error / 0.05236
            mode_fail = (slip > pos_tol) | (axis_error > angle_tol) | (wr[:, 1] < -0.1)
            success = pitch >= self.cfg.goal
            started = wr[:, 1] > 0.1
        self.last_gap = self.contact_gap(c, rot)
        separation = self.last_gap > self.cfg.gap_tolerance
        unloaded = self.normal_fraction < self.cfg.min_normal_load
        limits = (self.velocity.norm(dim=-1) > self.cfg.tray_speed_limit) | (
            self.omega.norm(dim=-1) > self.cfg.tray_angular_speed_limit
        )
        corners = self.bottom_corners(c, rot)
        # For tipping only the fixed +x edge must remain on the tray; the lifted
        # face may overhang. Full face contact uses all four projected vertices.
        support = corners[:, 2:] if task == "tipping" else corners
        outside = (support[:, :, :2].abs() > 0.4).any(dim=(1, 2))
        failed = mode_fail | separation | unloaded | limits | outside
        return success & ~failed, failed, started, error

    def bottom_corners(self, c, rot):
        corners = c.new_tensor([[-1, -1, -1], [-1, 1, -1], [1, -1, -1], [1, 1, -1]])
        return c[:, None] + torch.einsum(
            "nij,nkj->nki", rot, corners[None] * self.size[:, None] / 2
        )

    def contact_gap(self, c, rot):
        """Minimum bottom-corner gap; paired with load and mode-error checks."""
        return self.bottom_corners(c, rot)[:, :, 2].amin(1)

    def potential(self):
        """Shared goal shaping; terminal potential is zero in every group."""
        if self.cfg.task == "sticking":
            goal = self.position.new_tensor([self.cfg.goal, 0.0, 0.0])
            return -(self.position - goal).norm(dim=-1) / self.cfg.goal - 0.1 * self.velocity.norm(
                dim=-1
            )
        if self.cfg.task == "sliding":
            return (
                -(self.sim.data.qpos[:, 0] - self.origin[:, 0] - self.cfg.goal).abs()
                / self.cfg.goal
            )
        rot = rotation(self.sim.data.qpos[:, 3:7])
        angle = (
            torch.atan2(rot[:, 0, 2], rot[:, 0, 0])
            if self.cfg.task == "tipping"
            else torch.atan2(rot[:, 1, 0], rot[:, 0, 0])
        )
        return -(angle - self.cfg.goal).abs() / self.cfg.goal

    @torch.no_grad()
    def step(self, action, invalid=None):
        potential = self.potential()
        action = action.clamp(-1, 1)
        if invalid is None:
            invalid = torch.zeros_like(self.entered)
        # Invalid mappings terminate this transition. Zero is only a placeholder
        # for the batched simulator, never a claimed feasible fallback action.
        action = torch.where(invalid[:, None], 0.0, action)
        applied = action
        if self.cfg.disturbance:
            applied = (
                action
                + self.cfg.disturbance
                * torch.randn(action.shape, device=self.device, generator=self.noise_rng)
                * action.new_tensor(METRIC_SCALE)
                / self.scale
            ).clamp(-1, 1)
        margin, distance = self.zeros(), self.zeros()
        before_entry = ~self.entered
        if self.cfg.group in ("C", "D", "E"):
            startup = before_entry if self.cfg.group == "E" else torch.zeros_like(before_entry)
            inequalities, equalities = self.constraints(startup)
            margin, distance = measures(
                inequalities,
                equalities,
                applied,
                self.cfg.projection_iterations,
                action_scale=self.scale,
            )
        events = self.physics(applied * self.scale)
        self.steps += 1
        if events is None:
            success, failure, started, error = self.outcomes()
            duration = self.steps.float() * self.dt
        else:
            success, failure, started, error, duration = events
        success &= ~invalid
        physical_failure = failure & ~invalid
        failure |= invalid
        self.entered |= started
        timeout = self.steps >= self.max_steps
        done = success | failure | timeout
        reward = self.zeros() - self.dt
        reward += 10 * success.float() - 10 * ((failure | timeout) & ~success).float()
        reward += self.cfg.discount * (~done).float() * self.potential() - potential
        if self.cfg.group == "B":
            reward -= self.dt * self.cfg.conventional_weight * error
        if self.cfg.group in ("C", "D", "E"):
            physical = self.cfg.distance_weight * distance.square()
            if self.cfg.group in ("D", "E"):
                physical += self.cfg.margin_weight * torch.nn.functional.softplus(
                    (self.cfg.margin_reference - margin) / self.cfg.margin_temperature
                )
            if self.cfg.group == "E" and self.cfg.task != "sticking":
                physical *= torch.where(before_entry, self.cfg.startup_weight, 1.0)
            elif self.cfg.task != "sticking":
                physical *= (~before_entry).float()
            reward -= self.dt * physical
        info = dict(
            success=success.clone(),
            failure=failure.clone(),
            time=duration,
            margin=margin,
            distance=distance,
            mapping_failure=invalid.clone(),
            physical_failure=physical_failure,
            timeout=timeout & ~success & ~failure,
            **self.last_diagnostics,
        )
        # Return terminal outcomes while resetting completed worlds for the next action.
        ids = done.nonzero().flatten()
        if len(ids):
            self.reset(ids)
        return self.observe(), reward, done, info
