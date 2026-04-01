"""Tests for vehicle model."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
from landing_sim.vehicle import Vehicle, Leg


def test_leg_tip_position_upright():
    """Leg tip at zero pitch should be directly below CG."""
    leg = Leg("test", x_offset=0.0, z_offset=-25.0)
    x_tip, z_tip = leg.tip_position(0.0, 100.0, 0.0)
    assert abs(x_tip) < 1e-10
    assert abs(z_tip - 75.0) < 1e-10


def test_leg_tip_position_offset():
    """Leg with lateral offset should move with pitch."""
    leg = Leg("test", x_offset=4.0, z_offset=-25.0)
    x_tip, z_tip = leg.tip_position(0.0, 100.0, 0.0)
    assert abs(x_tip - 4.0) < 1e-10
    assert abs(z_tip - 75.0) < 1e-10


def test_leg_tip_position_pitched():
    """Leg tip should shift laterally when vehicle is pitched."""
    leg = Leg("test", x_offset=0.0, z_offset=-25.0)
    theta = np.radians(5)
    x_tip, z_tip = leg.tip_position(0.0, 100.0, theta)
    # Should have moved to the right (positive theta = tilt right)
    # x = 0 + 0*cos(theta) + (-25)*sin(theta) = -25*sin(5deg) ≈ -2.18
    # wait, sin(theta) is positive, so x = (-25)*sin(5°) ≈ -2.18
    # That means the base moves left when tilting right (which is correct - the base goes opposite to nose)
    assert x_tip < 0  # base goes left when nose tilts right


def test_vehicle_total_mass():
    v = Vehicle("test", 100000, 5000, 50, 9, 25, 2e7)
    assert v.total_mass == 105000


def test_leg_tip_velocity():
    leg = Leg("test", x_offset=4.0, z_offset=-25.0)
    vx, vz = leg.tip_velocity(0.0, -2.0, 0.0, 0.1)
    # At theta=0: vx = 0 + (-4*0 + (-25)*1)*0.1 = -2.5
    # vz = -2.0 + (-4*1 - (-25)*0)*0.1 = -2.0 + -0.4 = -2.4
    assert abs(vx - (-2.5)) < 1e-10
    assert abs(vz - (-2.4)) < 1e-10
