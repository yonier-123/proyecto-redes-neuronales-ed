"""
neural_field.py — Red neuronal que define el campo de velocidades del sistema.

Modelo matemático:
    dx/dt = f_θ(x, y) = tanh(w1·x + w2·y + b1)
    dy/dt = g_θ(x, y) = tanh(w3·x + w4·y + b2)

La red neuronal recibe la posición actual (x, y) y devuelve el vector
de velocidad (dx/dt, dy/dt). Los 6 parámetros θ = [w1, w2, w3, w4, b1, b2]
controlan completamente la dinámica del sistema.

Uso:
    >>> field = NeuralField([0.0, -1.0, 1.0, 0.0, 0.0, 0.0])
    >>> dx, dy = field.forward(1.0, 0.0)
    >>> print(f"Velocidad: ({dx:.4f}, {dy:.4f})")
    Velocidad: (0.0000, 0.7616)
"""

import numpy as np


class NeuralField:
    """
    Red neuronal simple de una capa que define un campo vectorial 2D.

    La red implementa dos funciones tanh que mapean posición → velocidad,
    creando un sistema dinámico continuo gobernado por una EDO.

    Attributes:
        weights (np.ndarray): Vector de 6 parámetros [w1, w2, w3, w4, b1, b2].
    """

    # Nombres legibles de cada peso (para visualización)
    WEIGHT_NAMES = ['w1', 'w2', 'w3', 'w4', 'b1', 'b2']
    NUM_WEIGHTS = 6

    def __init__(self, weights=None):
        """
        Inicializa la red neuronal con pesos dados o por defecto (círculo).

        Args:
            weights: Lista o array de 6 valores [w1, w2, w3, w4, b1, b2].
                     Si es None, usa pesos que generan una trayectoria circular.
        """
        if weights is not None:
            self.weights = np.array(weights, dtype=np.float64)
        else:
            # Pesos por defecto: generan una rotación (trayectoria circular)
            # dx/dt = tanh(-y) ≈ -y,   dy/dt = tanh(x) ≈ x
            self.weights = np.array([0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
                                    dtype=np.float64)

        if len(self.weights) != self.NUM_WEIGHTS:
            raise ValueError(
                f"Se esperan {self.NUM_WEIGHTS} pesos, "
                f"se recibieron {len(self.weights)}"
            )

    def forward(self, x, y):
        """
        Propagación hacia adelante: calcula el campo de velocidades.

        Calcula:
            dx/dt = tanh(w1·x + w2·y + b1)
            dy/dt = tanh(w3·x + w4·y + b2)

        Args:
            x: Coordenada x (escalar o array numpy para evaluación en malla).
            y: Coordenada y (escalar o array numpy para evaluación en malla).

        Returns:
            tuple: (dx/dt, dy/dt) — componentes del vector velocidad.
        """
        w1, w2, w3, w4, b1, b2 = self.weights
        dxdt = np.tanh(w1 * x + w2 * y + b1)
        dydt = np.tanh(w3 * x + w4 * y + b2)
        return dxdt, dydt

    def get_field(self, x_grid, y_grid):
        """
        Evalúa el campo vectorial sobre una malla 2D completa.

        Útil para visualizar el campo con streamplot() o quiver().

        Args:
            x_grid: Malla de coordenadas X (salida de np.meshgrid).
            y_grid: Malla de coordenadas Y (salida de np.meshgrid).

        Returns:
            tuple: (U, V) — componentes del campo vectorial en la malla.
        """
        return self.forward(x_grid, y_grid)

    def get_activations(self, x_grid, y_grid):
        """
        Calcula las activaciones pre-tanh de cada componente.

        Útil para los heatmaps de activaciones (Visualización 4).

        Args:
            x_grid: Malla de coordenadas X.
            y_grid: Malla de coordenadas Y.

        Returns:
            tuple: (act_x, act_y) — activaciones lineales antes de tanh.
        """
        w1, w2, w3, w4, b1, b2 = self.weights
        act_x = w1 * x_grid + w2 * y_grid + b1
        act_y = w3 * x_grid + w4 * y_grid + b2
        return act_x, act_y

    def set_weights(self, weights):
        """Actualiza los pesos de la red."""
        self.weights = np.array(weights, dtype=np.float64)

    def get_weights(self):
        """Devuelve una copia de los pesos actuales."""
        return self.weights.copy()

    def get_weights_dict(self):
        """Devuelve los pesos como diccionario {nombre: valor}."""
        return dict(zip(self.WEIGHT_NAMES, self.weights))

    def randomize_weights(self, scale=1.0, rng=None):
        """
        Inicializa los pesos con valores aleatorios.

        Args:
            scale: Escala de la distribución normal (desviación estándar).
            rng: Generador de números aleatorios numpy (opcional).
        """
        if rng is None:
            rng = np.random.default_rng()
        self.weights = rng.normal(0.0, scale, size=self.NUM_WEIGHTS)

    def __repr__(self):
        w = self.get_weights_dict()
        params = ", ".join(f"{k}={v:+.4f}" for k, v in w.items())
        return f"NeuralField({params})"
