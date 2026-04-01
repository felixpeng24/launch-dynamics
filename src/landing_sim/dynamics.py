"""Equations of motion: 3-DOF planar dynamics with contact and thrust."""

import numpy as np
from landing_sim.vehicle import Vehicle
from landing_sim.contact import ContactModel
from landing_sim.thrust import ThrustSystem


def state_derivatives(t: float, state: np.ndarray, vehicle: Vehicle,
                      contact: ContactModel, thrust_sys: ThrustSystem,
                      g: float, aero_config: dict | None = None,
                      wind_force: float = 0.0,
                      cg_offset: float = 0.0,
                      mass_scale: float = 1.0,
                      thrust_scale: float = 1.0,
                      leg_stiffness_scale: float = 1.0) -> np.ndarray:
    """Compute state derivatives for the 3-DOF landing dynamics.

    State vector: [z, z_dot, x, x_dot, theta, theta_dot]
        z     - vertical position of CG (altitude, m)
        z_dot - vertical velocity (m/s, positive up)
        x     - lateral position of CG (m)
        x_dot - lateral velocity (m/s)
        theta - pitch angle from vertical (rad, positive = tilted right)
        theta_dot - pitch rate (rad/s)
    """
    z, z_dot, x, x_dot, theta, theta_dot = state
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    mass = vehicle.total_mass * mass_scale
    inertia = vehicle.inertia_pitch * mass_scale

    # --- Gravity ---
    f_gravity_z = -mass * g

    # --- Thrust with TVC (Thrust Vector Control) ---
    thrust_mag, thrust_moment = thrust_sys.compute(z, z_dot, t)
    thrust_mag *= thrust_scale

    # PD controller for pitch via TVC gimbal
    # Gimbal angle offsets thrust direction to correct pitch errors
    kp_tvc = 5.0   # proportional gain (rad/rad)
    kd_tvc = 2.0   # derivative gain (rad/(rad/s))
    max_gimbal = np.radians(7.0)  # max gimbal angle (Raptor ~7 deg)
    gimbal = np.clip(-kp_tvc * theta - kd_tvc * theta_dot, -max_gimbal, max_gimbal)

    # Effective thrust direction = body axis + gimbal offset
    thrust_angle = theta + gimbal
    f_thrust_z = thrust_mag * np.cos(thrust_angle)
    f_thrust_x = thrust_mag * np.sin(thrust_angle)

    # Thrust moment: gimbal creates restoring moment, CG offset creates disturbance
    # M_gimbal = +T*sin(gimbal)*L (engine below CG: rightward gimbal → positive pitch moment)
    # M_cg = -T*cg_offset (CG right of thrust line → negative pitch moment)
    engine_arm = vehicle.cg_height  # distance from CG to engines (at base)
    m_thrust = (thrust_moment * thrust_scale
                + thrust_mag * np.sin(gimbal) * engine_arm
                - thrust_mag * cg_offset)

    # --- Aerodynamic drag (optional) ---
    f_aero_z = 0.0
    f_aero_x = 0.0
    if aero_config and aero_config.get("enabled", False):
        cd = aero_config["drag_coeff"]
        a_ref = aero_config["reference_area"]
        rho = aero_config["air_density"]
        q = 0.5 * rho
        # Drag opposes velocity
        v_mag_z = abs(z_dot)
        f_aero_z = -np.sign(z_dot) * q * cd * a_ref * v_mag_z * z_dot
        v_mag_x = abs(x_dot)
        f_aero_x = -np.sign(x_dot) * q * cd * a_ref * v_mag_x * x_dot

    # --- Wind ---
    f_wind_x = wind_force

    # --- Ground contact ---
    f_contact_z_total = 0.0
    f_contact_x_total = 0.0
    m_contact_total = 0.0

    # Create a modified contact model with scaled stiffness
    scaled_stiffness = contact.stiffness * leg_stiffness_scale

    for leg in vehicle.legs:
        x_tip, z_tip = leg.tip_position(x, z, theta)
        delta = -z_tip  # penetration depth (positive when below ground)

        if delta > 0:
            vx_tip, vz_tip = leg.tip_velocity(x_dot, z_dot, theta, theta_dot)
            delta_dot = -vz_tip  # rate of penetration

            # Compute forces with scaled stiffness
            f_spring = scaled_stiffness * delta ** contact.nonlinearity
            f_damper = contact.damping * delta_dot
            f_normal = max(f_spring + f_damper, 0.0)

            # Friction
            if abs(vx_tip) > 1e-6:
                f_friction = -contact.friction_coeff * f_normal * np.sign(vx_tip)
            else:
                f_friction = 0.0

            # Accumulate forces
            f_contact_z_total += f_normal
            f_contact_x_total += f_friction

            # Moment about CG: r x F
            rx = x_tip - x
            rz = z_tip - z
            m_contact_total += rx * f_normal - rz * f_friction

    # --- Sum forces ---
    f_z_total = f_gravity_z + f_thrust_z + f_aero_z + f_contact_z_total
    f_x_total = f_thrust_x + f_aero_x + f_wind_x + f_contact_x_total
    m_total = m_thrust + m_contact_total

    # --- Derivatives ---
    z_ddot = f_z_total / mass
    x_ddot = f_x_total / mass
    theta_ddot = m_total / inertia

    return np.array([z_dot, z_ddot, x_dot, x_ddot, theta_dot, theta_ddot])
