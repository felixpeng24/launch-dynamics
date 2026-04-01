"""Interface load extraction, statistical analysis, and load envelope computation."""

from dataclasses import dataclass
import numpy as np
from scipy.spatial import ConvexHull

from landing_sim.monte_carlo import MonteCarloResults


@dataclass
class LoadStatistics:
    """Statistical summary of a load quantity across Monte Carlo cases."""
    name: str
    unit: str
    mean: float
    std: float
    min_val: float
    max_val: float
    p50: float
    p95: float
    p99: float
    values: np.ndarray


@dataclass
class LoadEnvelope:
    """2D load envelope (e.g., axial vs. lateral)."""
    x_name: str
    y_name: str
    x_values: np.ndarray
    y_values: np.ndarray
    hull_vertices: np.ndarray | None = None


def compute_statistics(mc_results: MonteCarloResults) -> dict[str, LoadStatistics]:
    """Compute statistical summaries for all peak load quantities."""
    stats = {}

    load_data = {
        "peak_axial_force": ("Peak Axial Force", "N", mc_results.peak_axial),
        "peak_lateral_force": ("Peak Lateral Force", "N", mc_results.peak_lateral),
        "peak_moment": ("Peak Overturning Moment", "N-m", mc_results.peak_moment),
        "touchdown_vz": ("Touchdown Vertical Velocity", "m/s", mc_results.touchdown_vz),
        "touchdown_vx": ("Touchdown Lateral Velocity", "m/s", mc_results.touchdown_vx),
        "touchdown_theta": ("Touchdown Pitch Angle", "rad", mc_results.touchdown_theta),
    }

    for key, (name, unit, values) in load_data.items():
        stats[key] = LoadStatistics(
            name=name,
            unit=unit,
            mean=np.mean(values),
            std=np.std(values),
            min_val=np.min(values),
            max_val=np.max(values),
            p50=np.percentile(values, 50),
            p95=np.percentile(values, 95),
            p99=np.percentile(values, 99),
            values=values,
        )

    return stats


def compute_load_envelope(x_values: np.ndarray, y_values: np.ndarray,
                          x_name: str = "Axial", y_name: str = "Lateral") -> LoadEnvelope:
    """Compute 2D load envelope with convex hull."""
    points = np.column_stack([x_values, y_values])
    hull = None
    if len(points) >= 3:
        try:
            hull_obj = ConvexHull(points)
            hull = points[hull_obj.vertices]
        except Exception:
            hull = None
    return LoadEnvelope(
        x_name=x_name,
        y_name=y_name,
        x_values=x_values,
        y_values=y_values,
        hull_vertices=hull,
    )


def analytical_peak_load(mass: float, velocity: float, g: float,
                         stiffness: float, nonlinearity: float) -> float:
    """Estimate peak contact force using energy balance (impulse-momentum).

    For a simple vertical drop onto a nonlinear spring:
    KE = integral(F * d_delta) from 0 to delta_max
    0.5*m*v^2 + m*g*delta_max = k/(n+1) * delta_max^(n+1)

    This is solved iteratively for delta_max, then F_peak = k * delta_max^n.
    """
    # Iterative solution using Newton's method
    v = abs(velocity)
    ke = 0.5 * mass * v**2
    n = nonlinearity

    # Initial guess: linear spring
    delta_guess = (2 * ke / stiffness) ** (1 / (n + 1)) if stiffness > 0 else 0.01

    for _ in range(50):
        pe_spring = stiffness / (n + 1) * delta_guess ** (n + 1)
        pe_gravity = mass * g * delta_guess
        residual = pe_spring - ke - pe_gravity
        d_residual = stiffness * delta_guess ** n - mass * g

        if abs(d_residual) < 1e-20:
            break
        delta_new = delta_guess - residual / d_residual
        if delta_new < 0:
            delta_new = delta_guess / 2
        if abs(delta_new - delta_guess) < 1e-12:
            break
        delta_guess = delta_new

    return stiffness * delta_guess ** n


def format_statistics_table(stats: dict[str, LoadStatistics]) -> str:
    """Format statistics as a readable table string."""
    lines = []
    header = f"{'Quantity':<30} {'Mean':>12} {'Std':>12} {'P95':>12} {'P99':>12} {'Max':>12} {'Unit':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    for key, s in stats.items():
        lines.append(
            f"{s.name:<30} {s.mean:>12.1f} {s.std:>12.1f} "
            f"{s.p95:>12.1f} {s.p99:>12.1f} {s.max_val:>12.1f} {s.unit:>8}"
        )

    return "\n".join(lines)
