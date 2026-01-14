"""
Tests for functions in class SolveDiffusion2D
"""
import numpy as np
import pytest

from diffusion2d import SolveDiffusion2D


def test_initialize_domain():
    """
    Check function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()
    w = 20.
    h = 20.
    dx=0.2
    dy=0.2

    expected_nx = 100
    expected_ny = 100

    solver.initialize_domain(w,h,dx,dy)
    assert solver.nx == expected_nx, "value of nx is wrong, did the calculation change?"
    assert solver.ny == expected_ny, "value of ny is wrong, did the calculation change?"


def test_initialize_physical_parameters():
    """
    Checks function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()

    d = 5.
    T_cold = 350.
    T_hot = 750.
    solver.dx = 0.1
    solver.dy = 0.1

    expected_dt = 0.0005

    solver.initialize_physical_parameters(d, T_cold, T_hot)
    assert solver.dt == pytest.approx(expected_dt, abs=1e-4)


def test_set_initial_condition():
    """
    Checks function SolveDiffusion2D.get_initial_function
    """
    solver = SolveDiffusion2D()

    w = 20.
    h = 20.
    dx = 0.2
    dy = 0.2
    nx = int(w / dx)
    ny = int(h / dy)
    T_cold = 350.
    T_hot = 750.

    u = T_cold * np.ones((nx, ny))

    # Initial conditions - circle of radius r centred at (cx,cy) (mm)
    r, cx, cy = 2, 5, 5
    r2 = r ** 2
    for i in range(nx):
        for j in range(ny):
            p2 = (i * dx - cx) ** 2 + (j * dy - cy) ** 2
            if p2 < r2:
                u[i, j] = T_hot

    solver.T_cold = T_cold
    solver.T_hot = T_hot
    solver.w = w
    solver.h = h
    solver.dx = dx
    solver.dy = dy
    solver.nx = nx
    solver.ny = ny

    solver_u = solver.set_initial_condition()

    assert solver_u == pytest.approx(u, abs=1e-4)
