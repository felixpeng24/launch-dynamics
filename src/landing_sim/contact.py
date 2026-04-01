"""Nonlinear spring-damper ground contact model (Kelvin-Voigt)."""

from dataclasses import dataclass
import numpy as np


@dataclass
class ContactModel:
    stiffness: float      # N/m^n
    damping: float        # N*s/m
    nonlinearity: float   # exponent n
    friction_coeff: float  # Coulomb friction

    def compute_force(self, delta: float, delta_dot: float,
                      v_x_tip: float) -> tuple[float, float]:
        """Compute contact forces (vertical, lateral) for a single leg.

        Args:
            delta: Penetration depth (positive = below ground)
            delta_dot: Rate of penetration (positive = going deeper)
            v_x_tip: Lateral velocity of the leg tip

        Returns:
            (F_z, F_x): Vertical (upward +) and lateral contact forces
        """
        if delta <= 0:
            return 0.0, 0.0

        # Normal force (Kelvin-Voigt): spring + damper
        f_spring = self.stiffness * delta ** self.nonlinearity
        f_damper = self.damping * delta_dot
        f_normal = f_spring + f_damper

        # Ensure normal force is compressive only (no tensile ground contact)
        f_normal = max(f_normal, 0.0)

        # Friction force (Coulomb)
        if abs(v_x_tip) > 1e-6:
            f_friction = -self.friction_coeff * f_normal * np.sign(v_x_tip)
        else:
            f_friction = 0.0

        return f_normal, f_friction

    @classmethod
    def from_config(cls, config: dict) -> "ContactModel":
        c = config["contact"]
        return cls(
            stiffness=c["stiffness"],
            damping=c["damping"],
            nonlinearity=c["nonlinearity"],
            friction_coeff=c["friction_coeff"],
        )
