"""
Tests for functionality checks in class SolveDiffusion2D
"""

from diffusion2d import SolveDiffusion2D
import pytest
import numpy as np

def test_initialize_physical_parameters():
    """
    Checks function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()

    expected_dt = 0.0005

    solver.initialize_domain(10., 10., 0.1, 0.1)
    solver.initialize_physical_parameters(5.,350.,750.)

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

    solver.initialize_domain(w, h, dx, dy)
    solver.initialize_physical_parameters(5., 350., 750.)
    solver_u = solver.set_initial_condition()

    assert solver_u == pytest.approx(u, abs=1e-4)
