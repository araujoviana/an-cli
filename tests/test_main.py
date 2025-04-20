import pytest
import numpy as np
from an_cli.__main__ import simpson, differentiate, bisection, newton, lagrange, lsm

# Helper for floating point comparisons
approx = pytest.approx

# --- Simpson's Rule Tests ---


def test_simpson_simple_parabola():
    # Integral of x^2 from 0 to 1 is 1/3
    result = simpson(a=0.0, b=1.0, function="x**2")
    assert result == approx(1 / 3)


# --- Differentiation Tests ---


def test_differentiate_parabola_centered():
    # Derivative of x^2 at x=2 is 4
    result = differentiate(X=2.0, h=0.001, function="x**2", method="centered")
    assert result == approx(4.0)


# --- Bisection Tests ---


def test_bisection_simple_root():
    # Root of x**2 - 2 is sqrt(2)
    result = bisection(a=1.0, b=2.0, function="x**2 - 2", tolerance=1e-6)
    assert result == approx(np.sqrt(2), abs=1e-5)  # np.sqrt(2) is fine here


# --- Newton's Method Tests ---


def test_newton_simple_root():
    # Root of x**2 - 2 is sqrt(2)
    result = newton(estimate=1.5, function="x**2 - 2", tolerance=1e-7)
    assert result == approx(np.sqrt(2), abs=1e-6)  # np.sqrt(2) is fine here


# --- Lagrange Interpolation Tests ---


def test_lagrange_parabola():
    # Points from y = x^2: (1, 1), (2, 4), (3, 9)
    # Estimate at x = 2.5 should be 2.5^2 = 6.25
    result = lagrange(xs=[1.0, 2.0, 3.0], y=[1.0, 4.0, 9.0], r=2.5)
    assert result == approx(6.25)


# --- Least Squares Method Tests ---


def test_lsm_perfect_line():
    # Points exactly on y = 2x + 1: (1, 3), (2, 5), (3, 7)
    # Estimate at x = 2.5 should be 2*2.5 + 1 = 6
    result = lsm(xs=[1.0, 2.0, 3.0], y=[3.0, 5.0, 7.0], r=2.5)
    assert result == approx(6.0)
