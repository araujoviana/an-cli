import pytest
import numpy as np
from an_cli.__main__ import simpson, differentiate, bisection, newton, lagrange, lsm

# Helper function for floating point comparisons
approx = pytest.approx

# TODO other tests...


def test_lagrange_basic():
    xsarg = [0, 2, 4]
    yarg = [1, 5, 17]
    rarg = 10

    result = lagrange(xs=xsarg, y=yarg, r=rarg)
    expected = 101

    assert result == approx(expected, rel=1e-8)


def test_lagrange_three_points():
    xsarg = [1, 2, 3]
    yarg = [2, 4, 6]
    rarg = 2.5  # Interpolation for a point between 2 and 3

    result = lagrange(xs=xsarg, y=yarg, r=rarg)
    expected = (
        5  # Lagrange interpolation of these points should give a straight line result
    )

    assert result == approx(expected, rel=1e-8)


def test_lagrange_negative_values():
    xsarg = [-2, -1, 0]
    yarg = [4, 1, 0]
    rarg = -1.5  # Interpolation for a point between -2 and -1

    result = lagrange(xs=xsarg, y=yarg, r=rarg)
    expected = 2.25  # Result based on Lagrange interpolation

    assert result == approx(expected, rel=1e-8)


def test_lagrange_large_values():
    xsarg = [1e6, 2e6, 3e6]
    yarg = [2e6, 4e6, 6e6]
    rarg = 2.5e6  # Interpolation for a point between 2e6 and 3e6

    result = lagrange(xs=xsarg, y=yarg, r=rarg)
    expected = 5e6  # Should return the expected linear value

    assert result == approx(expected, rel=1e-8)


def test_lagrange_interpolation_outside_range():
    xsarg = [1, 2, 3]
    yarg = [1, 4, 9]
    rarg = 4  # Interpolation for a point outside the provided range

    result = lagrange(xs=xsarg, y=yarg, r=rarg)
    expected = 16  # The polynomial will continue the pattern, so expected value is 16

    assert result == approx(expected, rel=1e-8)


def test_lagrange_single_list():
    xsarg = [1]
    yarg = [2]
    rarg = 1

    # Single length lists are invalid
    with pytest.raises(SystemExit):
        lagrange(xs=xsarg, y=yarg, r=rarg)


# --- Least Squares Method Tests ---


def test_lsm_simple():
    xsarg = [1, 2, 3]
    yarg = [2, 4, 6]
    rarg = 5

    result = lsm(xs=xsarg, y=yarg, r=rarg)
    expected = 10

    assert result == approx(expected, rel=1e-8)


def test_lsm_complex():
    xsarg = [1, 2, 3, 4, 5]
    yarg = [5, 7, 9, 11, 13]
    rarg = 6

    result = lsm(xs=xsarg, y=yarg, r=rarg)
    expected = 15

    assert result == approx(expected, rel=1e-8)


def test_lsm_invalid_length():
    xsarg = [1]
    yarg = [2]
    rarg = 5

    with pytest.raises(SystemExit):
        lsm(xs=xsarg, y=yarg, r=rarg)


def test_lsm_empty_lists():
    xsarg = []
    yarg = []
    rarg = 5

    with pytest.raises(SystemExit):
        lsm(xs=xsarg, y=yarg, r=rarg)


def test_lsm_negative_values():
    xsarg = [-1, -2, -3]
    yarg = [-2, -4, -6]
    rarg = -4

    result = lsm(xs=xsarg, y=yarg, r=rarg)
    expected = -8

    assert result == approx(expected, rel=1e-8)


def test_lsm_random_points():
    xsarg = [0, 1, 2, 3, 4, 5]
    yarg = [0, 2, 4, 6, 8, 10]
    rarg = 7

    result = lsm(xs=xsarg, y=yarg, r=rarg)
    expected = 14  # This is a perfect linear relation y = 2x

    assert result == approx(expected, rel=1e-8)
