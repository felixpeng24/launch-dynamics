"""Vehicle model: mass properties, geometry, and landing leg configuration."""

from dataclasses import dataclass, field
import numpy as np
import yaml
from pathlib import Path


@dataclass
class Leg:
    name: str
    x_offset: float  # m, lateral offset from CG in body frame
    z_offset: float  # m, vertical offset from CG in body frame (negative = below)

    def tip_position(self, x_cg: float, z_cg: float, theta: float):
        """Compute leg tip position in world frame given vehicle state."""
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_world = x_cg + self.x_offset * cos_t + self.z_offset * sin_t
        z_world = z_cg - self.x_offset * sin_t + self.z_offset * cos_t
        return x_world, z_world

    def tip_velocity(self, x_dot: float, z_dot: float, theta: float, theta_dot: float):
        """Compute leg tip velocity in world frame."""
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_dot_world = x_dot + (-self.x_offset * sin_t + self.z_offset * cos_t) * theta_dot
        z_dot_world = z_dot + (-self.x_offset * cos_t - self.z_offset * sin_t) * theta_dot
        return x_dot_world, z_dot_world


@dataclass
class Vehicle:
    name: str
    dry_mass: float           # kg
    landing_propellant: float  # kg
    height: float             # m
    diameter: float           # m
    cg_height: float          # m above base
    inertia_pitch: float      # kg*m^2
    legs: list[Leg] = field(default_factory=list)

    @property
    def total_mass(self) -> float:
        return self.dry_mass + self.landing_propellant

    @classmethod
    def from_config(cls, config_path: str | Path) -> "Vehicle":
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        v = cfg["vehicle"]
        legs = [Leg(name=lg["name"], x_offset=lg["x_offset"], z_offset=lg["z_offset"])
                for lg in v["legs"]]
        return cls(
            name=v["name"],
            dry_mass=v["dry_mass"],
            landing_propellant=v["landing_propellant"],
            height=v["height"],
            diameter=v["diameter"],
            cg_height=v["cg_height"],
            inertia_pitch=v["inertia_pitch"],
            legs=legs,
        )
