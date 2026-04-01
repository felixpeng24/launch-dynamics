"""2D side-view landing animation using matplotlib."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch, Rectangle
from pathlib import Path

from landing_sim.simulation import SimResult


def create_landing_animation(result: SimResult, vehicle_height: float = 50.0,
                             vehicle_width: float = 9.0, leg_length: float = 5.0,
                             fps: int = 30, speedup: float = 1.0,
                             save_path: str | Path | None = None):
    """Create a 2D side-view animation of the landing.

    Args:
        result: Simulation result to animate.
        vehicle_height: Vehicle height in meters.
        vehicle_width: Vehicle width in meters.
        leg_length: Landing leg length in meters.
        fps: Frames per second for animation.
        speedup: Playback speed multiplier.
        save_path: Path to save animation (GIF or MP4).

    Returns:
        FuncAnimation object.
    """
    # Subsample for smooth playback
    total_time = result.t[-1] - result.t[0]
    n_frames = int(total_time * fps / speedup)
    frame_times = np.linspace(result.t[0], result.t[-1], n_frames)

    # Interpolate state
    z_interp = np.interp(frame_times, result.t, result.z)
    x_interp = np.interp(frame_times, result.t, result.x)
    theta_interp = np.interp(frame_times, result.t, result.theta)

    # Figure setup
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(-80, 80)
    ax.set_ylim(-10, max(z_interp[0] + vehicle_height, 100) * 1.1)
    ax.set_aspect("equal")
    ax.set_xlabel("Lateral Position (m)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title("Landing Dynamics — Side View")

    # Ground
    ax.axhline(0, color="#8B7355", linewidth=3, zorder=1)
    ax.fill_between([-100, 100], [-10, -10], [0, 0], color="#D2B48C", alpha=0.3, zorder=0)

    # Landing pad
    pad_width = 30
    ax.fill_between([-pad_width/2, pad_width/2], [-0.5, -0.5], [0, 0],
                    color="#555555", alpha=0.8, zorder=1)

    # Vehicle body (will be updated each frame)
    hw = vehicle_width / 2
    hh = vehicle_height / 2

    body_patch = plt.Polygon([[0, 0]], closed=True, facecolor="#C0C0C0",
                              edgecolor="#333333", linewidth=2, zorder=5)
    ax.add_patch(body_patch)

    # Nose cone
    nose_patch = plt.Polygon([[0, 0]], closed=True, facecolor="#A0A0A0",
                              edgecolor="#333333", linewidth=2, zorder=5)
    ax.add_patch(nose_patch)

    # Legs
    leg_lines = [ax.plot([], [], color="#444444", linewidth=2, zorder=4)[0] for _ in range(2)]

    # Thrust plume
    plume_patch = plt.Polygon([[0, 0]], closed=True, facecolor="#FF6600",
                               edgecolor="#FF3300", alpha=0.7, linewidth=1, zorder=3)
    ax.add_patch(plume_patch)

    # Info text
    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                        fontsize=10, verticalalignment="top",
                        fontfamily="monospace",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    def _rotate(points, theta, cx, cz):
        """Rotate points around (cx, cz) by angle theta."""
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        px = points[:, 0] - cx
        pz = points[:, 1] - cz
        rx = px * cos_t + pz * sin_t + cx
        rz = -px * sin_t + pz * cos_t + cz
        return np.column_stack([rx, rz])

    def update(frame):
        x_cg = x_interp[frame]
        z_cg = z_interp[frame]
        theta = theta_interp[frame]
        t = frame_times[frame]

        # Body rectangle vertices (in body frame, CG at origin)
        body_verts = np.array([
            [-hw, -hh],
            [hw, -hh],
            [hw, hh * 0.7],
            [-hw, hh * 0.7],
        ])
        # Translate to CG position, then rotate
        body_verts[:, 0] += x_cg
        body_verts[:, 1] += z_cg
        body_verts = _rotate(body_verts, theta, x_cg, z_cg)
        body_patch.set_xy(body_verts)

        # Nose cone
        nose_verts = np.array([
            [-hw, hh * 0.7],
            [hw, hh * 0.7],
            [0, hh],
        ])
        nose_verts[:, 0] += x_cg
        nose_verts[:, 1] += z_cg
        nose_verts = _rotate(nose_verts, theta, x_cg, z_cg)
        nose_patch.set_xy(nose_verts)

        # Landing legs
        leg_base_l = np.array([-hw * 0.8, -hh]) + np.array([x_cg, z_cg])
        leg_base_r = np.array([hw * 0.8, -hh]) + np.array([x_cg, z_cg])
        leg_tip_l = np.array([-hw * 1.5, -hh - leg_length]) + np.array([x_cg, z_cg])
        leg_tip_r = np.array([hw * 1.5, -hh - leg_length]) + np.array([x_cg, z_cg])

        for pts in [leg_base_l, leg_base_r, leg_tip_l, leg_tip_r]:
            rotated = _rotate(pts.reshape(1, 2), theta, x_cg, z_cg)
            pts[0], pts[1] = rotated[0, 0], rotated[0, 1]

        leg_lines[0].set_data([leg_base_l[0], leg_tip_l[0]], [leg_base_l[1], leg_tip_l[1]])
        leg_lines[1].set_data([leg_base_r[0], leg_tip_r[0]], [leg_base_r[1], leg_tip_r[1]])

        # Thrust plume (visible when above ground)
        if z_cg > 0.5:
            plume_len = max(5, min(30, abs(z_cg) * 0.1))
            plume_width = hw * 0.6
            plume_verts = np.array([
                [-plume_width, -hh],
                [plume_width, -hh],
                [plume_width * 0.3, -hh - plume_len],
                [-plume_width * 0.3, -hh - plume_len],
            ])
            plume_verts[:, 0] += x_cg
            plume_verts[:, 1] += z_cg
            plume_verts = _rotate(plume_verts, theta, x_cg, z_cg)
            plume_patch.set_xy(plume_verts)
            plume_patch.set_alpha(0.7)
        else:
            plume_patch.set_alpha(0.0)

        # Info text
        vz = np.interp(t, result.t, result.z_dot)
        vx = np.interp(t, result.t, result.x_dot)
        info_text.set_text(
            f"t = {t:.2f} s\n"
            f"Alt = {z_cg:.1f} m\n"
            f"Vz = {vz:.2f} m/s\n"
            f"Vx = {vx:.2f} m/s\n"
            f"Pitch = {np.degrees(theta):.2f} deg"
        )

        return [body_patch, nose_patch, plume_patch, info_text] + leg_lines

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000 / fps, blit=True)

    if save_path:
        save_path = Path(save_path)
        if save_path.suffix == ".gif":
            anim.save(str(save_path), writer="pillow", fps=fps)
        else:
            anim.save(str(save_path), writer="ffmpeg", fps=fps)

    return anim
