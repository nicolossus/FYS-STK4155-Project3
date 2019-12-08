#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def diffusion_solver(I, alpha=1, L=1, Nx=11, T=1, F=0.5, udim='1D'):
    """
    Solver for 1D diffusion equations with constant diffusion coefficent and
    no source term:
                            u_t = a*u_xx
    on spatial domain (0,L) and temporal domain [0, T].

    Parameters
    ----------
    I : Initial condition, I(x), as callable object (i.e. a function)
    alpha : Diffusion coefficient
    L : Length of the spatial domain ([0,L])
    Nx : Number of mesh cells; mesh points are numbered from 0 to Nx
    T : Simulation (stop) time
    F : The dimensionless number a*dt/dx**2; implicitly specifies the time step
    udim : '1D' or '2D' solution array as output
    """

    assert alpha > 0, f'Diffusion coefficient alpha must be greater than 0'
    assert F <= 0.5, \
        f'Stability criterion F=alpha*dt/dx**2 <= 0.5 not satisfied with F={F}'
    x = np.linspace(0, L, Nx + 1)       # mesh points in space
    dx = x[1] - x[0]                    # constant mesh spacing in x
    dt = F * dx**2 / alpha              # constant mesh spacing in t
    Nt = int(T / dt)
    t = np.linspace(0, T, Nt + 1)        # mesh points in time
    u_arr = np.zeros((Nt + 1, Nx + 1))   # solution array
    u = np.zeros(Nx + 1)                 # 1D solution array
    u1 = I(x)                            # initial condition
    u1[0] = u1[Nx] = 0.0                 # boundary conditions
    u_arr[0] = u1

    for n in range(0, Nt):
        # Update all inner points
        u[1:Nx] = u1[1:Nx] + F * (u1[0:Nx - 1] - 2 * u1[1:Nx] + u1[2:Nx + 1])
        # Boundary conditions
        u[0] = 0.0
        u[Nx] = 0.0
        u_arr[n] = u
        # Update u1 before next step
        u1, u = u, u1  # just switch references

    if udim == '2D':
        return u_arr, x, t
    else:
        return u, x, t


if __name__ == "__main__":
    def I(x):
        """
        Initial condition
        """

        return np.sin(np.pi * x)

    def u_exact(x, t):
        """
        Analytic solution
        """

        return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)

    u, x, t = diffusion_solver(I, alpha=1, L=1, Nx=11, T=1, F=0.5, udim='2D')
    x, t = np.meshgrid(x, t)
    ue = u_exact(x, t)

    diff = np.abs(ue - u)
    print(f"Max diff: {np.max(diff)}")
    print(f"Mean diff: {np.mean(diff)}")

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.set_title("Analytic")
    ax.plot_surface(x, t, ue)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.set_title("Forward Euler")
    ax.plot_surface(x, t, u)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.set_title("Diff")
    ax.plot_surface(x, t, diff)

    plt.show()
