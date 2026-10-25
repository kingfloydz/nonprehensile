from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg

from .geometry import SCALE, acceleration_constraints, measures
from .math import mv, quat_step, rotation

XML = """
<mujoco model="accelerating_tray">
  <option timestep="0.001" gravity="0 0 0" cone="elliptic"/>
  <default><geom condim="3" solref="0.004 1" solimp="0.95 0.99 0.001"/></default>
  <worldbody>
    <geom name="tray" type="box" pos="0 0 -0.01" size="0.4 0.4 0.01"
          friction="0.8 0 0" rgba="0.25 0.4 0.55 1"/>
    <body name="box" pos="0 0 0.1">
      <freejoint/>
      <geom name="payload" type="box" size="0.1 0.1 0.1" density="500"
            priority="1" friction="0.5 0 0" rgba="0.85 0.5 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""


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
    discount: float = 0.99
    translation_tolerance: float = 0.01
    rotation_tolerance: float = float(np.deg2rad(10))
    tray_speed_limit: float = 5.0
    tray_angular_speed_limit: float = 10.0


class TrayEnv:
    """MuJoCo-Warp simulation in the accelerating tray frame.

    A fixed tray and an independently simulated payload avoid a servo or an
    artificially heavy tray. Exact non-inertial force/torque terms implement
    commanded tray acceleration. No artificial contact or mode enforcement.
    """

    dt = 0.02
    substeps = 20

    def __init__(self, cfg, bank, device="cuda:0"):
        self.cfg, self.device, self.n = cfg, device, cfg.num_envs
        self.rng = torch.Generator(device=device).manual_seed(cfg.seed)
        self.noise_rng = torch.Generator(device=device).manual_seed(cfg.seed + 1000)
        with np.load(bank) as data:
            self.bank = {
                k: torch.tensor(data[k], device=device)
                for k in ("size", "mu", cfg.task + "_h", cfg.task + "_e")
            }
        model = mujoco.MjModel.from_xml_string(XML)
        self.sim = Simulation(
            self.n,
            SimulationCfg(
                nconmax=16,
                njmax=64,
                mujoco=MujocoCfg(
                    timestep=0.001,
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
        self.scale = torch.tensor(SCALE, device=device)
        self.all_ids = torch.arange(self.n, device=device)
        self.reset(self.all_ids)

    def zeros(self, *shape):
        return torch.zeros((self.n, *shape), device=self.device)

    def reset(self, ids):
        count = len(ids)
        indices = torch.randint(
            len(self.bank["size"]), (count,), device=self.device, generator=self.rng
        )
        self.bank_ids[ids] = indices
        size = self.bank["size"][indices]
        mass = 500 * size.prod(-1)
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
        qpos[ids, :3] = 0
        qpos[ids, 2] = size[:, 2] / 2
        qpos[ids, 3:7] = qpos.new_tensor([1.0, 0, 0, 0])
        if self.cfg.task in ("sliding", "tipping"):
            qpos[ids, 0] = -0.1
        self.origin[ids] = qpos[ids, :3]
        self.origin[ids, 2] = 0
        self.position[ids], self.velocity[ids], self.omega[ids] = 0, 0, 0
        self.quaternion[ids] = qpos.new_tensor([1.0, 0, 0, 0])
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
        if self.cfg.task == "sticking":
            goal = mv(rt, c.new_tensor([0.5, 0, 0]) - self.position) / 0.5
        elif self.cfg.task == "sliding":
            goal[:, 0] = (self.origin[:, 0] + 0.2 - c[:, 0]) / 0.2
        else:
            angle = (
                torch.atan2(r[:, 0, 2], r[:, 0, 0])
                if self.cfg.task == "tipping"
                else torch.atan2(r[:, 1, 0], r[:, 0, 0])
            )
            goal[:, 0] = 1 - angle / (torch.pi / 2)
        low, high = c.new_tensor([0.06, 0.06, 0.05]), c.new_tensor([0.30, 0.30, 0.40])
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
                2 * (self.mu[:, None] - 0.1) / 0.7 - 1,
                goal,
                self.entered[:, None].float(),
                self.steps[:, None] / 250,
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
            self.bank[task + "_h"][ids],
            self.bank[task + "_e"][ids],
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
        # Tipping ends at first new-face contact, before post-impact rebound.
        # Latch the first terminal event (failure wins simultaneous events).
        event_success = torch.zeros_like(self.entered)
        event_failure = torch.zeros_like(self.entered)
        event_started = torch.zeros_like(self.entered)
        event_error = self.zeros()
        event_time = (self.steps + 1).float() * self.dt
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
            self.position += self.velocity * 0.001 + world_accel * 0.0000005
            self.velocity += world_accel * 0.001
            self.quaternion = quat_step(self.quaternion, omega + alpha * 0.0005, 0.001)
            self.omega += alpha * 0.001
            self.sim.step()
            if self.cfg.task == "tipping":
                success, failure, started, error = self.outcomes()
                active = ~(event_success | event_failure)
                first = active & (success | failure)
                event_error = torch.where(active, error, event_error)
                event_started |= active & started
                event_success |= first & success
                event_failure |= first & failure
                event_time = torch.where(
                    first, self.steps.float() * self.dt + (substep + 1) * 0.001, event_time
                )
        if self.cfg.task == "tipping":
            return event_success, event_failure, event_started, event_error, event_time
        return None

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
            target = c.new_tensor([0.5, 0.0, 0.0])
            success = (self.position - target).norm(dim=-1) < 0.005
            success &= (self.velocity.norm(dim=-1) < 0.03) & (self.omega.norm(dim=-1) < 0.1)
            started = torch.ones_like(success)
        elif task == "sliding":
            error = delta[:, 1].abs() / 0.003 + angle / 0.05236
            mode_fail = (delta[:, 1].abs() > pos_tol) | (angle > angle_tol) | (vr[:, 0] < -0.03)
            success = delta[:, 0] >= 0.2
            started = vr[:, 0] > 0.02
        elif task == "twisting":
            error = delta[:, :2].norm(dim=-1) / 0.003 + tilt / 0.05236
            mode_fail = (
                (delta[:, :2].norm(dim=-1) > pos_tol) | (tilt > angle_tol) | (wr[:, 2] < -0.1)
            )
            success = yaw >= torch.pi / 2
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
            local[:, 2] *= -1
            next_edge = c + mv(rot, local)
            success = (pitch > torch.pi / 2 - 0.05236) & (next_edge[:, 2].abs() < 0.002)
            started = wr[:, 1] > 0.1
        separation = self.contact_gap(c, rot) > 0.008
        limits = (self.velocity.norm(dim=-1) > self.cfg.tray_speed_limit) | (
            self.omega.norm(dim=-1) > self.cfg.tray_angular_speed_limit
        )
        outside = (c[:, :2].abs() > 0.4).any(1)
        failed = mode_fail | separation | limits | outside
        return success & ~failed, failed, started, error

    def contact_gap(self, c, rot):
        """Minimum original bottom-corner height; diagnostic, not a force test."""
        corners = c.new_tensor([[-1, -1, -1], [-1, 1, -1], [1, -1, -1], [1, 1, -1]])
        bottom = c[:, None] + torch.einsum(
            "nij,nkj->nki", rot, corners[None] * self.size[:, None] / 2
        )
        return bottom[:, :, 2].amin(1)

    def potential(self):
        """Shared goal shaping; terminal potential is zero in every group."""
        if self.cfg.task == "sticking":
            goal = self.position.new_tensor([0.5, 0.0, 0.0])
            return -(self.position - goal).norm(dim=-1) / 0.5 - 0.1 * self.velocity.norm(dim=-1)
        if self.cfg.task == "sliding":
            return -(self.sim.data.qpos[:, 0] - self.origin[:, 0] - 0.2).abs() / 0.2
        rot = rotation(self.sim.data.qpos[:, 3:7])
        angle = (
            torch.atan2(rot[:, 0, 2], rot[:, 0, 0])
            if self.cfg.task == "tipping"
            else torch.atan2(rot[:, 1, 0], rot[:, 0, 0])
        )
        return -(angle - torch.pi / 2).abs() / (torch.pi / 2)

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
            ).clamp(-1, 1)
        margin, distance = self.zeros(), self.zeros()
        before_entry = ~self.entered
        if self.cfg.group in ("C", "D", "E"):
            startup = before_entry if self.cfg.group == "E" else torch.zeros_like(before_entry)
            inequalities, equalities = self.constraints(startup)
            margin, distance = measures(
                inequalities, equalities, applied, self.cfg.projection_iterations
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
        timeout = self.steps >= 250
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
            mode_error=error.clone(),
            contact_gap=self.contact_gap(*self.state()[:2]),
        )
        # Return terminal outcomes while resetting completed worlds for the next action.
        ids = done.nonzero().flatten()
        if len(ids):
            self.reset(ids)
        return self.observe(), reward, done, info
