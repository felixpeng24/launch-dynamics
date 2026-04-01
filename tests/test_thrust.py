"""Tests for thrust models."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
from landing_sim.thrust import SimpleThrustProfile, MultiEngineThrustProfile, Engine, ThrustSystem


def test_simple_no_thrust_above_burn_altitude():
    profile = SimpleThrustProfile(
        max_thrust=2.2e6, min_throttle=0.4,
        burn_start_altitude=500, target_velocity=-1.0)
    assert profile.compute(600, -50, 0) == 0.0


def test_simple_thrust_at_burn_start():
    profile = SimpleThrustProfile(
        max_thrust=2.2e6, min_throttle=0.4,
        burn_start_altitude=500, target_velocity=-1.0)
    thrust = profile.compute(500, -50, 0)
    assert thrust > 0
    assert thrust <= 2.2e6


def test_simple_thrust_cutoff_at_touchdown():
    """Thrust should be zero at or below touchdown CG altitude."""
    profile = SimpleThrustProfile(
        max_thrust=2.2e6, min_throttle=0.4,
        burn_start_altitude=500, target_velocity=-1.0,
        touchdown_cg_altitude=25.0)
    assert profile.compute(25.0, -1, 0) == 0.0
    assert profile.compute(10.0, -1, 0) == 0.0
    # Above touchdown but below burn start: should have thrust
    assert profile.compute(100.0, -20.0, 0) > 0.0


def test_multi_engine_total_thrust():
    engines = [
        Engine(x_offset=-1.5, thrust=2.2e6, min_throttle=0.4),
        Engine(x_offset=0.0, thrust=2.2e6, min_throttle=0.4),
        Engine(x_offset=1.5, thrust=2.2e6, min_throttle=0.4),
    ]
    profile = MultiEngineThrustProfile(engines=engines, burn_start_altitude=500)
    total, moment = profile.compute(250, -20, 0)
    assert total > 0
    # Symmetric engines → moment should be zero (or very close)
    assert abs(moment) < 1.0


def test_multi_engine_shutdown():
    engines = [
        Engine(x_offset=-1.5, thrust=2.2e6, min_throttle=0.4, shutdown_altitude=50),
        Engine(x_offset=0.0, thrust=2.2e6, min_throttle=0.4),
        Engine(x_offset=1.5, thrust=2.2e6, min_throttle=0.4, shutdown_altitude=50),
    ]
    profile = MultiEngineThrustProfile(engines=engines, burn_start_altitude=500)

    # Above shutdown altitude: all engines on
    t1, _ = profile.compute(100, -10, 0)

    # Below shutdown altitude: only center engine
    t2, m2 = profile.compute(40, -5, 0)

    assert t2 < t1  # fewer engines = less thrust
    assert abs(m2) < 1.0  # center engine only → no moment
