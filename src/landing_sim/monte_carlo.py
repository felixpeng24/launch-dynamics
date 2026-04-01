"""Monte Carlo batch simulation with Latin Hypercube Sampling."""

from dataclasses import dataclass, field
import numpy as np
from scipy.stats.qmc import LatinHypercube
from scipy.stats import norm
from multiprocessing import Pool, cpu_count
from functools import partial

from landing_sim.simulation import run_simulation, SimResult


@dataclass
class MonteCarloSample:
    """Dispersed parameters for a single Monte Carlo case."""
    vertical_velocity: float
    lateral_velocity: float
    pitch_angle: float
    pitch_rate: float
    thrust_scale: float
    mass_scale: float
    cg_offset: float
    wind_force: float
    leg_stiffness_scale: float


@dataclass
class MonteCarloResults:
    """Aggregated Monte Carlo results."""
    n_cases: int
    samples: list[MonteCarloSample]
    results: list[SimResult]
    peak_axial: np.ndarray = field(default_factory=lambda: np.array([]))
    peak_lateral: np.ndarray = field(default_factory=lambda: np.array([]))
    peak_moment: np.ndarray = field(default_factory=lambda: np.array([]))
    touchdown_vz: np.ndarray = field(default_factory=lambda: np.array([]))
    touchdown_vx: np.ndarray = field(default_factory=lambda: np.array([]))
    touchdown_theta: np.ndarray = field(default_factory=lambda: np.array([]))
    settled_flags: np.ndarray = field(default_factory=lambda: np.array([]))


def generate_samples(config: dict, n_cases: int | None = None,
                     seed: int | None = None) -> list[MonteCarloSample]:
    """Generate Latin Hypercube samples for Monte Carlo dispersions."""
    mc_cfg = config["monte_carlo"]
    n = n_cases or mc_cfg["n_cases"]
    rng_seed = seed if seed is not None else mc_cfg.get("seed", 42)

    dispersions = mc_cfg["dispersions"]
    param_names = list(dispersions.keys())
    n_params = len(param_names)

    # Latin Hypercube Sampling
    sampler = LatinHypercube(d=n_params, seed=rng_seed)
    lhs_unit = sampler.random(n=n)

    # Transform uniform [0,1] samples to Gaussian with specified mean/sigma
    samples = []
    for i in range(n):
        vals = {}
        for j, name in enumerate(param_names):
            nominal = dispersions[name]["nominal"]
            sigma = dispersions[name]["sigma"]
            vals[name] = norm.ppf(lhs_unit[i, j], loc=nominal, scale=sigma)
        samples.append(MonteCarloSample(**vals))

    return samples


def _run_single_case(args: tuple) -> SimResult:
    """Run a single Monte Carlo case (for multiprocessing)."""
    config, sample = args
    ic_overrides = {
        "vertical_velocity": sample.vertical_velocity,
        "lateral_velocity": sample.lateral_velocity,
        "pitch_angle": sample.pitch_angle,
        "pitch_rate": sample.pitch_rate,
    }
    return run_simulation(
        config,
        ic_overrides=ic_overrides,
        thrust_scale=sample.thrust_scale,
        mass_scale=sample.mass_scale,
        cg_offset=sample.cg_offset,
        wind_force=sample.wind_force,
        leg_stiffness_scale=sample.leg_stiffness_scale,
    )


def run_monte_carlo(config: dict, n_cases: int | None = None,
                    seed: int | None = None,
                    n_workers: int | None = None,
                    progress_callback=None) -> MonteCarloResults:
    """Run full Monte Carlo campaign.

    Args:
        config: Full configuration dictionary.
        n_cases: Number of cases (overrides config).
        seed: Random seed (overrides config).
        n_workers: Number of parallel workers. None = cpu_count.
        progress_callback: Optional callable(completed, total) for progress updates.

    Returns:
        MonteCarloResults with all case results and extracted peak loads.
    """
    samples = generate_samples(config, n_cases, seed)
    n = len(samples)

    workers = n_workers or max(1, cpu_count() - 1)

    # Run cases
    results = []
    if workers == 1:
        for i, sample in enumerate(samples):
            result = _run_single_case((config, sample))
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, n)
    else:
        args_list = [(config, s) for s in samples]
        with Pool(processes=workers) as pool:
            for i, result in enumerate(pool.imap(_run_single_case, args_list)):
                results.append(result)
                if progress_callback:
                    progress_callback(i + 1, n)

    # Extract peak loads
    peak_axial = np.zeros(n)
    peak_lateral = np.zeros(n)
    peak_moment = np.zeros(n)
    touchdown_vz = np.zeros(n)
    touchdown_vx = np.zeros(n)
    touchdown_theta = np.zeros(n)
    settled_flags = np.zeros(n, dtype=bool)

    for i, (result, sample) in enumerate(zip(results, samples)):
        peak_axial[i] = np.max(result.total_forces.get("axial", [0]))
        peak_lateral[i] = np.max(np.abs(result.total_forces.get("lateral", [0])))
        peak_moment[i] = np.max(np.abs(result.total_forces.get("moment", [0])))

        # Touchdown conditions (at first contact)
        td_idx = result.touchdown_idx or 0
        touchdown_vz[i] = result.z_dot[td_idx]
        touchdown_vx[i] = result.x_dot[td_idx]
        touchdown_theta[i] = result.theta[td_idx]
        settled_flags[i] = result.settled

    return MonteCarloResults(
        n_cases=n,
        samples=samples,
        results=results,
        peak_axial=peak_axial,
        peak_lateral=peak_lateral,
        peak_moment=peak_moment,
        touchdown_vz=touchdown_vz,
        touchdown_vx=touchdown_vx,
        touchdown_theta=touchdown_theta,
        settled_flags=settled_flags,
    )
