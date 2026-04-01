"""Tests for contact model."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
from landing_sim.contact import ContactModel


@pytest.fixture
def contact():
    return ContactModel(stiffness=5e8, damping=1e6, nonlinearity=1.5, friction_coeff=0.5)


def test_no_contact_when_above_ground(contact):
    fz, fx = contact.compute_force(delta=-0.1, delta_dot=0.0, v_x_tip=0.0)
    assert fz == 0.0
    assert fx == 0.0


def test_no_contact_at_surface(contact):
    fz, fx = contact.compute_force(delta=0.0, delta_dot=0.0, v_x_tip=0.0)
    assert fz == 0.0
    assert fx == 0.0


def test_normal_force_positive_penetration(contact):
    fz, fx = contact.compute_force(delta=0.01, delta_dot=0.0, v_x_tip=0.0)
    assert fz > 0  # upward force
    expected = 5e8 * 0.01**1.5
    assert abs(fz - expected) < 1e-3


def test_damping_adds_to_force(contact):
    fz_nodamp, _ = contact.compute_force(delta=0.01, delta_dot=0.0, v_x_tip=0.0)
    fz_damp, _ = contact.compute_force(delta=0.01, delta_dot=1.0, v_x_tip=0.0)
    assert fz_damp > fz_nodamp


def test_friction_opposes_motion(contact):
    _, fx = contact.compute_force(delta=0.01, delta_dot=0.0, v_x_tip=1.0)
    assert fx < 0  # friction opposes positive velocity

    _, fx2 = contact.compute_force(delta=0.01, delta_dot=0.0, v_x_tip=-1.0)
    assert fx2 > 0  # friction opposes negative velocity


def test_no_tensile_force(contact):
    # Large negative delta_dot (rebounding fast) could make total force negative
    fz, _ = contact.compute_force(delta=0.001, delta_dot=-1e6, v_x_tip=0.0)
    assert fz >= 0
