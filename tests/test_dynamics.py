"""Tests for dynamics equations of motion."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
from landing_sim.vehicle import Vehicle, Leg
from landing_sim.contact import ContactModel
from landing_sim.thrust import ThrustSystem, SimpleThrustProfile
from landing_sim.dynamics import state_derivatives


@pytest.fixture
def simple_vehicle():
    return Vehicle(
        name="test",
        dry_mass=1000,
        landing_propellant=0,
        height=10,
        diameter=2,
        cg_height=5,
        inertia_pitch=10000,
        legs=[Leg("left", -1.0, -5.0), Leg("right", 1.0, -5.0)],
    )


@pytest.fixture
def contact():
    return ContactModel(stiffness=1e7, damping=1e4, nonlinearity=1.5, friction_coeff=0.3)


@pytest.fixture
def thrust_off():
    simple = SimpleThrustProfile(0, 0.4, 500, -1.0)
    return ThrustSystem(mode="simple", simple=simple)


def test_free_fall(simple_vehicle, contact, thrust_off):
    """Vehicle in free fall should accelerate at -g."""
    state = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    derivs = state_derivatives(0, state, simple_vehicle, contact, thrust_off, 9.81)
    # z_ddot should be -g (no contact, no thrust at high altitude)
    assert abs(derivs[1] - (-9.81)) < 1e-6
    assert abs(derivs[3]) < 1e-10  # no lateral acceleration
    assert abs(derivs[5]) < 1e-10  # no pitch acceleration


def test_contact_produces_upward_force(simple_vehicle, contact, thrust_off):
    """When legs penetrate ground, upward force should be generated."""
    # Place vehicle so legs are just below ground (CG at 4.9m, legs at -0.1m)
    state = np.array([4.9, -1.0, 0.0, 0.0, 0.0, 0.0])
    derivs = state_derivatives(0, state, simple_vehicle, contact, thrust_off, 9.81)
    # z_ddot should be > -g (contact force reduces downward acceleration)
    assert derivs[1] > -9.81


def test_symmetric_no_rotation(simple_vehicle, contact, thrust_off):
    """Symmetric impact should produce no pitch acceleration."""
    state = np.array([4.9, -1.0, 0.0, 0.0, 0.0, 0.0])
    derivs = state_derivatives(0, state, simple_vehicle, contact, thrust_off, 9.81)
    assert abs(derivs[5]) < 1e-6
