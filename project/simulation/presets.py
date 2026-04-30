"""
presets.py — Configuraciones predefinidas de parámetros.

Cada preset contiene los pesos de la red neuronal y la condición inicial
que generan un tipo específico de trayectoria. Esto permite demostrar
distintos comportamientos del sistema sin ajustar manualmente.

Presets disponibles:
    circulo         — Rotación pura (trayectoria circular)
    espiral_entrada — Espiral convergente al origen
    espiral_salida  — Espiral divergente desde el origen
    punto_fijo      — Convergencia directa al origen (nodo estable)
    oscilacion      — Comportamiento oscilatorio amortiguado
    caos            — Comportamiento complejo no trivial

Uso:
    >>> from simulation.presets import get_preset, list_presets
    >>> print(list_presets())
    ['circulo', 'espiral_entrada', 'espiral_salida', ...]
    >>> preset = get_preset('espiral_entrada')
    >>> print(preset['descripcion'])
    Espiral convergente al origen
"""


PRESETS = {
    'circulo': {
        'weights': [0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
        'x0': 1.0,
        'y0': 0.0,
        'descripcion': 'Rotación pura — trayectoria circular',
        'color': '#00d4ff',
    },
    'espiral_entrada': {
        'weights': [-0.1, -1.0, 1.0, -0.1, 0.0, 0.0],
        'x0': 1.5,
        'y0': 0.0,
        'descripcion': 'Espiral convergente al origen',
        'color': '#7c4dff',
    },
    'espiral_salida': {
        'weights': [0.1, -1.0, 1.0, 0.1, 0.0, 0.0],
        'x0': 0.1,
        'y0': 0.0,
        'descripcion': 'Espiral divergente desde el origen',
        'color': '#ff6b6b',
    },
    'punto_fijo': {
        'weights': [-1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
        'x0': 1.0,
        'y0': 1.0,
        'descripcion': 'Convergencia al origen (nodo estable)',
        'color': '#00ff88',
    },
    'oscilacion': {
        'weights': [-0.2, -0.8, 0.8, -0.2, 0.1, -0.1],
        'x0': 1.2,
        'y0': 0.5,
        'descripcion': 'Oscilación amortiguada',
        'color': '#ffab40',
    },
    'caos': {
        'weights': [0.5, -1.2, 1.3, 0.3, 0.2, -0.1],
        'x0': 0.5,
        'y0': 0.5,
        'descripcion': 'Comportamiento complejo no trivial',
        'color': '#e040fb',
    },
}


def get_preset(name):
    """
    Devuelve un preset por nombre.

    Args:
        name (str): Nombre del preset.

    Returns:
        dict: Diccionario con 'weights', 'x0', 'y0', 'descripcion', 'color'.

    Raises:
        KeyError: Si el preset no existe.
    """
    if name not in PRESETS:
        available = ', '.join(PRESETS.keys())
        raise KeyError(
            f"Preset '{name}' no encontrado. Disponibles: {available}"
        )
    return PRESETS[name].copy()


def list_presets():
    """Devuelve la lista de nombres de presets disponibles."""
    return list(PRESETS.keys())


def get_preset_info():
    """
    Devuelve información resumida de todos los presets.

    Returns:
        list[dict]: Lista de dicts con 'name' y 'descripcion'.
    """
    return [
        {'name': name, 'descripcion': p['descripcion']}
        for name, p in PRESETS.items()
    ]
