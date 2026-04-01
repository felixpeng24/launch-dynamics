"""Publication-quality static plots for simulation results."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pathlib import Path

from landing_sim.simulation import SimResult
from landing_sim.monte_carlo import MonteCarloResults
from landing_sim.loads import LoadStatistics, LoadEnvelope

# Style configuration
COLORS = {
    "primary": "#1a5276",
    "secondary": "#c0392b",
    "accent": "#27ae60",
    "neutral": "#7f8c8d",
    "highlight": "#f39c12",
}

def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.dpi": 150,
    })


def plot_time_history(result: SimResult, save_path: str | Path | None = None):
    """Plot time histories for a single simulation case."""
    setup_style()
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), tight_layout=True)
    fig.suptitle("Landing Dynamics — Time History", fontsize=14, fontweight="bold")

    # Altitude
    ax = axes[0, 0]
    ax.plot(result.t, result.z, color=COLORS["primary"], linewidth=1.5)
    ax.set_ylabel("Altitude (m)")
    ax.set_title("CG Altitude")

    # Vertical velocity
    ax = axes[0, 1]
    ax.plot(result.t, result.z_dot, color=COLORS["primary"], linewidth=1.5)
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Vertical Velocity")

    # Lateral position
    ax = axes[1, 0]
    ax.plot(result.t, result.x, color=COLORS["secondary"], linewidth=1.5)
    ax.set_ylabel("Position (m)")
    ax.set_title("Lateral Position")

    # Lateral velocity
    ax = axes[1, 1]
    ax.plot(result.t, result.x_dot, color=COLORS["secondary"], linewidth=1.5)
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Lateral Velocity")

    # Pitch angle
    ax = axes[2, 0]
    ax.plot(result.t, np.degrees(result.theta), color=COLORS["accent"], linewidth=1.5)
    ax.set_ylabel("Angle (deg)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Pitch Angle")

    # Contact forces
    ax = axes[2, 1]
    if "axial" in result.total_forces:
        ax.plot(result.t, result.total_forces["axial"] / 1e6,
                color=COLORS["primary"], linewidth=1.5, label="Axial")
        ax.plot(result.t, np.abs(result.total_forces["lateral"]) / 1e6,
                color=COLORS["secondary"], linewidth=1.5, label="|Lateral|")
        ax.legend(fontsize=9)
    ax.set_ylabel("Force (MN)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Contact Forces")

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_contact_per_leg(result: SimResult, leg_names: list[str] | None = None,
                         save_path: str | Path | None = None):
    """Plot contact forces for each individual leg."""
    setup_style()
    if "leg_fz" not in result.contact_forces:
        return None

    leg_fz = result.contact_forces["leg_fz"]
    n_legs = leg_fz.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), tight_layout=True)
    fig.suptitle("Per-Leg Contact Forces", fontsize=14, fontweight="bold")

    cmap = plt.cm.tab10
    for j in range(n_legs):
        label = leg_names[j] if leg_names else f"Leg {j}"
        axes[0].plot(result.t, leg_fz[:, j] / 1e6, linewidth=1, label=label, color=cmap(j))

    axes[0].set_ylabel("Normal Force (MN)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_title("Vertical Contact Force per Leg")
    axes[0].legend(fontsize=8, ncol=2)

    # Total
    axes[1].plot(result.t, result.total_forces["axial"] / 1e6,
                 color=COLORS["primary"], linewidth=2)
    axes[1].set_ylabel("Force (MN)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Total Axial Contact Force")

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_mc_histograms(mc_results: MonteCarloResults,
                       save_path: str | Path | None = None):
    """Plot histograms of peak loads from Monte Carlo analysis."""
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), tight_layout=True)
    fig.suptitle(f"Monte Carlo Peak Loads — {mc_results.n_cases} Cases",
                 fontsize=14, fontweight="bold")

    data = [
        (mc_results.peak_axial / 1e6, "Peak Axial Force (MN)", COLORS["primary"]),
        (mc_results.peak_lateral / 1e6, "Peak Lateral Force (MN)", COLORS["secondary"]),
        (mc_results.peak_moment / 1e6, "Peak Moment (MN-m)", COLORS["accent"]),
    ]

    for ax, (values, label, color) in zip(axes, data):
        ax.hist(values, bins=50, color=color, alpha=0.75, edgecolor="white", linewidth=0.5)
        ax.axvline(np.mean(values), color="black", linestyle="--",
                   linewidth=1.5, label=f"Mean: {np.mean(values):.2f}")
        ax.axvline(np.percentile(values, 95), color=COLORS["highlight"], linestyle="--",
                   linewidth=1.5, label=f"P95: {np.percentile(values, 95):.2f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_load_envelope(envelope: LoadEnvelope, save_path: str | Path | None = None):
    """Plot 2D load envelope with convex hull."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)

    ax.scatter(envelope.x_values / 1e6, envelope.y_values / 1e6,
               s=5, alpha=0.3, color=COLORS["primary"], label="Cases")

    if envelope.hull_vertices is not None:
        hull = np.vstack([envelope.hull_vertices, envelope.hull_vertices[0]])
        ax.plot(hull[:, 0] / 1e6, hull[:, 1] / 1e6,
                color=COLORS["secondary"], linewidth=2, label="Convex Hull")

    ax.set_xlabel(f"{envelope.x_name} (MN)")
    ax.set_ylabel(f"{envelope.y_name} (MN)")
    ax.set_title("Load Envelope")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_scatter_colored(mc_results: MonteCarloResults,
                         save_path: str | Path | None = None):
    """Scatter plot: touchdown velocity vs peak axial force, colored by pitch angle."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)

    scatter = ax.scatter(
        mc_results.touchdown_vz,
        mc_results.peak_axial / 1e6,
        c=np.degrees(mc_results.touchdown_theta),
        s=8, alpha=0.6, cmap="RdYlBu_r",
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Touchdown Pitch Angle (deg)")

    ax.set_xlabel("Touchdown Vertical Velocity (m/s)")
    ax.set_ylabel("Peak Axial Force (MN)")
    ax.set_title("Touchdown Velocity vs Peak Load")

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_cdf(values: np.ndarray, label: str, unit: str,
             save_path: str | Path | None = None):
    """Plot cumulative distribution function for a load quantity."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)

    sorted_vals = np.sort(values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

    ax.plot(sorted_vals, cdf, color=COLORS["primary"], linewidth=2)
    ax.axhline(0.95, color=COLORS["highlight"], linestyle="--", alpha=0.7, label="95th percentile")
    ax.axhline(0.99, color=COLORS["secondary"], linestyle="--", alpha=0.7, label="99th percentile")
    ax.set_xlabel(f"{label} ({unit})")
    ax.set_ylabel("CDF")
    ax.set_title(f"CDF — {label}")
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig
