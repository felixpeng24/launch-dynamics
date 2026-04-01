"""Thrust profile models: simple throttle-down and multi-engine configurations."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SimpleThrustProfile:
    """Single equivalent thrust with constant-deceleration hoverslam guidance."""
    max_thrust: float
    min_throttle: float
    burn_start_altitude: float
    target_velocity: float  # desired vertical velocity at touchdown
    mass: float = 105000.0  # kg, set from config
    gravity: float = 9.80665
    touchdown_cg_altitude: float = 25.0  # m, CG height at leg contact

    def compute(self, altitude: float, velocity_z: float, t: float) -> float:
        """Compute thrust using constant-deceleration guidance (hoverslam).

        Targets reaching target_velocity when CG reaches touchdown_cg_altitude
        (i.e., when legs contact ground).
        """
        if altitude > self.burn_start_altitude:
            return 0.0

        # Cut engines at/below touchdown altitude (contact detected)
        if altitude <= self.touchdown_cg_altitude:
            return 0.0

        # Distance remaining to leg contact (positive value)
        z_remaining = max(altitude - self.touchdown_cg_altitude, 0.1)
        v_current = velocity_z  # negative = descending
        v_target = self.target_velocity

        # Kinematic equation: v_f^2 = v_i^2 + 2*a*(-z_remaining)
        # where -z_remaining is the displacement (downward = negative)
        # Solving for a (positive = upward): a = (v_i^2 - v_f^2) / (2*z_remaining)
        a_required = (v_current**2 - v_target**2) / (2.0 * z_remaining)

        # Thrust = m * (g + a_required)
        thrust_required = self.mass * (self.gravity + a_required)

        # Clamp to achievable range
        thrust = np.clip(thrust_required, self.max_thrust * self.min_throttle, self.max_thrust)
        return float(thrust)


@dataclass
class Engine:
    x_offset: float           # m, lateral position relative to vehicle centerline
    thrust: float             # N, max thrust
    min_throttle: float
    shutdown_altitude: float | None = None  # m, altitude at which this engine shuts down
    active: bool = True

    def compute(self, altitude: float, throttle_command: float) -> float:
        if not self.active:
            return 0.0
        if self.shutdown_altitude is not None and altitude <= self.shutdown_altitude:
            self.active = False
            return 0.0
        return self.thrust * max(self.min_throttle, throttle_command)


@dataclass
class MultiEngineThrustProfile:
    """Multiple engines with individual throttle curves and sequential shutdown."""
    engines: list[Engine] = field(default_factory=list)
    burn_start_altitude: float = 500.0
    mass: float = 105000.0
    gravity: float = 9.80665

    def compute(self, altitude: float, velocity_z: float, t: float) -> tuple[float, float]:
        """Compute total thrust and net moment about vehicle centerline.

        Returns:
            (total_thrust, moment): Total thrust magnitude and pitch moment from asymmetry
        """
        if altitude > self.burn_start_altitude:
            return 0.0, 0.0

        # Compute required throttle using guidance
        td_alt = getattr(self, '_touchdown_alt', 25.0)
        z_remaining = max(altitude - td_alt, 0.1)
        v_target = -1.0
        a_required = (velocity_z**2 - v_target**2) / (2.0 * z_remaining)
        n_active = sum(1 for e in self.engines if e.active and
                       (e.shutdown_altitude is None or altitude > e.shutdown_altitude))
        n_active = max(n_active, 1)
        thrust_per_eng = self.mass * (self.gravity + a_required) / n_active
        max_per_eng = self.engines[0].thrust if self.engines else 2.2e6
        min_per_eng = max_per_eng * (self.engines[0].min_throttle if self.engines else 0.4)
        throttle_cmd = np.clip(thrust_per_eng / max_per_eng, 0.4, 1.0)

        total_thrust = 0.0
        moment = 0.0
        for engine in self.engines:
            f = engine.compute(altitude, throttle_cmd)
            total_thrust += f
            moment += f * engine.x_offset  # moment = force * arm

        return total_thrust, moment

    def reset(self):
        for engine in self.engines:
            engine.active = True


@dataclass
class ThrustSystem:
    """Unified thrust interface supporting both simple and multi-engine modes."""
    mode: str  # "simple" or "multi_engine"
    simple: SimpleThrustProfile | None = None
    multi_engine: MultiEngineThrustProfile | None = None

    def compute(self, altitude: float, velocity_z: float, t: float) -> tuple[float, float]:
        """Returns (total_thrust, thrust_moment)."""
        if self.mode == "simple":
            return self.simple.compute(altitude, velocity_z, t), 0.0
        else:
            return self.multi_engine.compute(altitude, velocity_z, t)

    def reset(self):
        if self.multi_engine:
            self.multi_engine.reset()

    @classmethod
    def from_config(cls, config: dict) -> "ThrustSystem":
        tc = config["thrust"]
        mode = tc["mode"]

        vehicle_mass = (config["vehicle"]["dry_mass"] +
                        config["vehicle"]["landing_propellant"])
        gravity = config.get("gravity", 9.80665)

        cg_height = config["vehicle"]["cg_height"]
        simple = SimpleThrustProfile(
            max_thrust=tc["simple"]["max_thrust"],
            min_throttle=tc["simple"]["min_throttle"],
            burn_start_altitude=tc["simple"]["burn_start_altitude"],
            target_velocity=tc["simple"]["target_velocity"],
            mass=vehicle_mass,
            gravity=gravity,
            touchdown_cg_altitude=cg_height,
        )

        engines = []
        me_cfg = tc["multi_engine"]
        for i, ep in enumerate(me_cfg["engine_positions"]):
            shutdown_alt = None
            for seq in me_cfg.get("shutdown_sequence", []):
                if seq["engine_idx"] == i:
                    shutdown_alt = seq["shutdown_altitude"]
            engines.append(Engine(
                x_offset=ep["x_offset"],
                thrust=me_cfg["thrust_per_engine"],
                min_throttle=me_cfg["min_throttle"],
                shutdown_altitude=shutdown_alt,
            ))
        multi = MultiEngineThrustProfile(
            engines=engines,
            burn_start_altitude=me_cfg["burn_start_altitude"],
            mass=vehicle_mass,
            gravity=gravity,
        )
        multi._touchdown_alt = cg_height

        return cls(mode=mode, simple=simple, multi_engine=multi)
