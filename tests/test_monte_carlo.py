"""Tests for Monte Carlo sampling."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
from landing_sim.monte_carlo import generate_samples
from landing_sim.simulation import load_config


@pytest.fixture
def config():
    return load_config()


def test_sample_count(config):
    samples = generate_samples(config, n_cases=50, seed=42)
    assert len(samples) == 50


def test_sample_reproducibility(config):
    s1 = generate_samples(config, n_cases=10, seed=123)
    s2 = generate_samples(config, n_cases=10, seed=123)
    for a, b in zip(s1, s2):
        assert a.vertical_velocity == b.vertical_velocity
        assert a.thrust_scale == b.thrust_scale


def test_sample_distribution(config):
    """Check that samples roughly follow the specified distribution."""
    samples = generate_samples(config, n_cases=5000, seed=42)

    vz_values = [s.vertical_velocity for s in samples]
    mean_vz = np.mean(vz_values)
    std_vz = np.std(vz_values)

    nominal = config["monte_carlo"]["dispersions"]["vertical_velocity"]["nominal"]
    sigma = config["monte_carlo"]["dispersions"]["vertical_velocity"]["sigma"]

    assert abs(mean_vz - nominal) < 3 * sigma / np.sqrt(5000)
    assert abs(std_vz - sigma) < 0.1 * sigma


def test_lhs_coverage(config):
    """LHS should cover the parameter space more uniformly than random."""
    samples = generate_samples(config, n_cases=100, seed=42)
    thrust_scales = sorted([s.thrust_scale for s in samples])

    # Check that we don't have large gaps (LHS property)
    diffs = np.diff(thrust_scales)
    max_gap = np.max(diffs)
    mean_gap = np.mean(diffs)
    assert max_gap < 25 * mean_gap  # LHS gaps should be bounded
