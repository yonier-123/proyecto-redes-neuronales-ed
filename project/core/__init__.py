"""
core/ — Núcleo matemático del Neural ODE Simulator.

Módulos:
    neural_field: Red neuronal que define el campo de velocidades (EDO).
    ode_solver:   Solvers numéricos (Euler, RK4) para integrar la EDO.
    trainer:      Descenso de gradiente para entrenar la red.
"""

from .neural_field import NeuralField
from .ode_solver import ODESolver

__all__ = ['NeuralField', 'ODESolver']
