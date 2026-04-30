"""
simulation/ — Configuraciones y algoritmos de entrenamiento.

Módulos:
    presets:     Configuraciones predefinidas de parámetros (círculo, espiral, etc.)
    optimizers:  Algoritmos de optimización (SGD, Momentum, Adam).
"""

from .presets import get_preset, list_presets

__all__ = ['get_preset', 'list_presets']
