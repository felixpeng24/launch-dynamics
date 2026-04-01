"""ODE integration and single-case simulation runner."""

from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import solve_ivp
import yaml
from pathlib import Path

from landing_sim.vehicle import Vehicle
from landing_sim.contact import ContactModel
from landing_sim.thrust import ThrustSystem
from landing_sim.dynamics import state_derivatives


@dataclass
class SimResult:
    """Container for single simulation results."""
    t: np.ndarray              # time vector
    z: np.ndarray              # altitude
    z_dot: np.ndarray          # vertical velocity
    x: np.ndarray              # lateral position
    x_dot: np.ndarray          # lateral velocity
    theta: np.ndarray          # pitch angle
    theta_dot: np.ndarray      # pitch rate
    contact_forces: dict = field(default_factory=dict)  # per-leg forces over time
    total_forces: dict = field(default_factory=dict)     # total interface loads
    settled: bool = False
    touchdown_idx: int | None = None

    @property
    def state_array(self) -> np.ndarray:
        return np.column_stack([self.z, self.z_dot, self.x, self.x_dot,
                                self.theta, self.theta_dot])


def _compute_contact_forces_post(result: SimResult, vehicle: Vehicle,
                                 contact: ContactModel,
                                 leg_stiffness_scale: float = 1.0) -> None:
    """Post-process to extract per-leg and total contact forces at each time step."""
    n_steps = len(result.t)
    n_legs = len(vehicle.legs)

    leg_fz = np.zeros((n_steps, n_legs))
    leg_fx = np.zeros((n_steps, n_legs))
    total_fz = np.zeros(n_steps)
    total_fx = np.zeros(n_steps)
    total_moment = np.zeros(n_steps)

    scaled_stiffness = contact.stiffness * leg_stiffness_scale

    for i in range(n_steps):
        z_i = result.z[i]
        x_i = result.x[i]
        theta_i = result.theta[i]
        z_dot_i = result.z_dot[i]
        x_dot_i = result.x_dot[i]
        theta_dot_i = result.theta_dot[i]

        for j, leg in enumerate(vehicle.legs):
            x_tip, z_tip = leg.tip_position(x_i, z_i, theta_i)
            delta = -z_tip

            if delta > 0:
                vx_tip, vz_tip = leg.tip_velocity(x_dot_i, z_dot_i, theta_i, theta_dot_i)
                delta_dot = -vz_tip

                f_spring = scaled_stiffness * delta ** contact.nonlinearity
                f_damper = contact.damping * delta_dot
                f_normal = max(f_spring + f_damper, 0.0)

                if abs(vx_tip) > 1e-6:
                    f_friction = -contact.friction_coeff * f_normal * np.sign(vx_tip)
                else:
                    f_friction = 0.0

                leg_fz[i, j] = f_normal
                leg_fx[i, j] = f_friction
                total_fz[i] += f_normal
                total_fx[i] += f_friction

                rx = x_tip - x_i
                rz = z_tip - z_i
                total_moment[i] += rx * f_normal - rz * f_friction

    result.contact_forces = {
        "leg_fz": leg_fz,
        "leg_fx": leg_fx,
    }
    result.total_forces = {
        "axial": total_fz,
        "lateral": total_fx,
        "moment": total_moment,
    }


def run_simulation(config: dict,
                   ic_overrides: dict | None = None,
                   thrust_scale: float = 1.0,
                   mass_scale: float = 1.0,
                   cg_offset: float = 0.0,
                   wind_force: float = 0.0,
                   leg_stiffness_scale: float = 1.0) -> SimResult:
    """Run a single landing simulation.

    Args:
        config: Full configuration dictionary (from YAML).
        ic_overrides: Override initial conditions. Keys match config IC names.
        thrust_scale: Multiplicative thrust uncertainty factor.
        mass_scale: Multiplicative mass uncertainty factor.
        cg_offset: Lateral CG offset in meters.
        wind_force: Constant lateral wind force in Newtons.
        leg_stiffness_scale: Multiplicative leg stiffness factor.

    Returns:
        SimResult with full time history.
    """
    vehicle = Vehicle.from_config_dict(config)
    contact_model = ContactModel.from_config(config)
    thrust_sys = ThrustSystem.from_config(config)
    thrust_sys.reset()

    g = config["gravity"]
    sim_cfg = config["simulation"]
    aero_cfg = config.get("aerodynamics")

    # Initial conditions
    ic = config["initial_conditions"].copy()
    if ic_overrides:
        ic.update(ic_overrides)

    y0 = np.array([
        ic["altitude"],
        ic["vertical_velocity"],
        ic["lateral_position"],
        ic["lateral_velocity"],
        ic["pitch_angle"],
        ic["pitch_rate"],
    ])

    # Event: vehicle settled (low velocity after contact)
    first_contact_time = [None]

    def settled_event(t, y):
        z, z_dot, x, x_dot, theta, theta_dot = y
        # Check if any leg is in contact
        in_contact = False
        for leg in vehicle.legs:
            _, z_tip = leg.tip_position(x, z, theta)
            if z_tip <= 0:
                in_contact = True
                break
        if not in_contact:
            return 1.0  # not settled

        if first_contact_time[0] is None:
            first_contact_time[0] = t

        elapsed = t - first_contact_time[0]
        vert_speed = abs(z_dot) + abs(theta_dot) * 10.0  # weighted pitch rate
        if elapsed > sim_cfg["settle_time"] and vert_speed < sim_cfg["settle_velocity"]:
            return -1.0  # settled
        return 1.0

    settled_event.terminal = True
    settled_event.direction = -1

    # Event: vehicle tipped over
    def tipover_event(t, y):
        return np.pi / 4 - abs(y[4])  # 45 degrees = tipped over

    tipover_event.terminal = True
    tipover_event.direction = -1

    def rhs(t, y):
        return state_derivatives(
            t, y, vehicle, contact_model, thrust_sys, g,
            aero_config=aero_cfg,
            wind_force=wind_force,
            cg_offset=cg_offset,
            mass_scale=mass_scale,
            thrust_scale=thrust_scale,
            leg_stiffness_scale=leg_stiffness_scale,
        )

    sol = solve_ivp(
        rhs,
        [sim_cfg["t_start"], sim_cfg["t_max"]],
        y0,
        method=sim_cfg["method"],
        max_step=sim_cfg["dt_max"],
        rtol=sim_cfg["rtol"],
        atol=sim_cfg["atol"],
        events=[settled_event, tipover_event],
        dense_output=True,
    )

    # Find touchdown index
    touchdown_idx = None
    for i, z_val in enumerate(sol.y[0]):
        in_contact = False
        for leg in vehicle.legs:
            _, z_tip = leg.tip_position(sol.y[2, i], z_val, sol.y[4, i])
            if z_tip <= 0:
                in_contact = True
                break
        if in_contact:
            touchdown_idx = i
            break

    result = SimResult(
        t=sol.t,
        z=sol.y[0],
        z_dot=sol.y[1],
        x=sol.y[2],
        x_dot=sol.y[3],
        theta=sol.y[4],
        theta_dot=sol.y[5],
        settled=len(sol.t_events[0]) > 0 if sol.t_events else False,
        touchdown_idx=touchdown_idx,
    )

    # Post-process contact forces
    _compute_contact_forces_post(result, vehicle, contact_model, leg_stiffness_scale)

    return result


def load_config(config_path: str | Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# Add from_config_dict to Vehicle for use without file
def _vehicle_from_config_dict(config: dict) -> Vehicle:
    v = config["vehicle"]
    from landing_sim.vehicle import Leg
    legs = [Leg(name=lg["name"], x_offset=lg["x_offset"], z_offset=lg["z_offset"])
            for lg in v["legs"]]
    return Vehicle(
        name=v["name"],
        dry_mass=v["dry_mass"],
        landing_propellant=v["landing_propellant"],
        height=v["height"],
        diameter=v["diameter"],
        cg_height=v["cg_height"],
        inertia_pitch=v["inertia_pitch"],
        legs=legs,
    )


# Monkey-patch for convenience
Vehicle.from_config_dict = staticmethod(lambda config: _vehicle_from_config_dict(config))
