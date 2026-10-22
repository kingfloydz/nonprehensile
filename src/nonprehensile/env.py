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
    <geom name="tray" type="box" pos="0 0 -0.01" size="0.3 0.3 0.01"
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
    disturbance: float = 0.0
    discount: float = 0.99


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
        self.previous_action = self.zeros(6)
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
        self.steps[ids], self.entered[ids], self.previous_action[ids] = 0, False, 0
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
        return torch.cat(
            (
                c - self.origin,
                r.flatten(1),
                vr,
                wr,
                self.position,
                self.velocity,
                self.omega,
                self.quaternion,
                g / 9.81,
                self.size,
                self.mu[:, None],
                self.entered[:, None].float(),
                self.previous_action,
                self.steps[:, None] / 250,
            ),
            -1,
        )

    def physics(self, chi):
        alpha, accel = chi[:, :3], chi[:, 3:]
        for _ in range(self.substeps):
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

    def outcomes(self):
        c, rot, vr, wr, _, _ = self.state()
        delta = c - self.origin
        task = self.cfg.task
        angle = torch.acos(((rot.diagonal(dim1=1, dim2=2).sum(1) - 1) / 2).clamp(-1, 1))
        tilt = torch.acos(rot[:, 2, 2].clamp(-1, 1))
        yaw = torch.atan2(rot[:, 1, 0], rot[:, 0, 0])
        pitch = torch.atan2(rot[:, 0, 2], rot[:, 0, 0])
        if task == "sticking":
            error = delta[:, :2].norm(dim=-1) / 0.003 + angle / 0.05236
            mode_fail = (delta[:, :2].norm(dim=-1) > 0.003) | (angle > 0.05236)
            target = c.new_tensor([0.5, 0.0, 0.0])
            success = (self.position - target).norm(dim=-1) < 0.005
            success &= (self.velocity.norm(dim=-1) < 0.03) & (self.omega.norm(dim=-1) < 0.1)
            started = torch.ones_like(success)
        elif task == "sliding":
            error = delta[:, 1].abs() / 0.003 + angle / 0.05236
            mode_fail = (delta[:, 1].abs() > 0.003) | (angle > 0.05236) | (vr[:, 0] < -0.03)
            success = delta[:, 0] >= 0.2
            started = vr[:, 0] > 0.02
        elif task == "twisting":
            error = delta[:, :2].norm(dim=-1) / 0.003 + tilt / 0.05236
            mode_fail = (delta[:, :2].norm(dim=-1) > 0.003) | (tilt > 0.05236) | (wr[:, 2] < -0.1)
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
            mode_fail = (slip > 0.003) | (axis_error > 0.05236) | (wr[:, 1] < -0.1)
            local[:, 2] *= -1
            next_edge = c + mv(rot, local)
            success = (pitch > torch.pi / 2 - 0.05236) & (next_edge[:, 2].abs() < 0.002)
            started = wr[:, 1] > 0.1
        corners = c.new_tensor([[-1, -1, -1], [-1, 1, -1], [1, -1, -1], [1, 1, -1]])
        bottom = c[:, None] + torch.einsum(
            "nij,nkj->nki", rot, corners[None] * self.size[:, None] / 2
        )
        separation = bottom[:, :, 2].amin(1) > 0.008
        limits = (self.velocity.norm(dim=-1) > 2.0) | (self.omega.norm(dim=-1) > 3.0)
        outside = (c[:, :2].abs() > 0.3).any(1)
        failed = mode_fail | separation | limits | outside
        return success & ~failed, failed, started, error

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
    def step(self, action):
        potential = self.potential()
        action = action.clamp(-1, 1)
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
            c, rot, vr, wr, gravity, inertia = self.state()
            ids = self.bank_ids
            startup = before_entry if self.cfg.group == "E" else torch.zeros_like(before_entry)
            h, e = self.bank[self.cfg.task + "_h"][ids], self.bank[self.cfg.task + "_e"][ids]
            inequalities, equalities = acceleration_constraints(
                self.cfg.task,
                h,
                e,
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
            )
            margin, distance = measures(
                inequalities, equalities, applied, self.cfg.projection_iterations
            )
        self.physics(applied * self.scale)
        self.steps += 1
        success, failure, started, error = self.outcomes()
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
        self.previous_action.copy_(action)
        info = dict(
            success=success.clone(),
            failure=failure.clone(),
            time=self.steps.float() * self.dt,
            margin=margin,
            distance=distance,
        )
        # Return terminal outcomes while resetting completed worlds for the next action.
        ids = done.nonzero().flatten()
        if len(ids):
            self.reset(ids)
        return self.observe(), reward, done, info
