"""
optimizers.py — Algoritmos de optimización para el descenso de gradiente.

Implementa tres optimizadores clásicos que actualizan los pesos θ
de la red neuronal durante el entrenamiento:

    SGD:      W ← W - η·∇L            (el más simple)
    Momentum: v ← μ·v - η·∇L, W ← W + v  (agrega inercia)
    Adam:     Combina momentum + RMSprop  (el más robusto)

Cada optimizador implementa la interfaz:
    optimizer.update(weights, gradients) → new_weights
    optimizer.reset()                   → reinicia estado interno

El descenso de gradiente en tiempo continuo es la EDO:
    dW/dt = -η · ∇L(W)
Los optimizadores son distintas discretizaciones de esta EDO.
"""

import numpy as np


class SGD:
    """
    Descenso de gradiente estocástico (Stochastic Gradient Descent).

    La discretización más simple de dW/dt = -η·∇L:
        W_{t+1} = W_t - η · ∇L(W_t)
    """

    name = 'SGD'

    def __init__(self, lr=0.01):
        """
        Args:
            lr (float): Tasa de aprendizaje η.
        """
        self.lr = lr

    def update(self, weights, gradients):
        """Actualiza los pesos con un paso de SGD."""
        return weights - self.lr * gradients

    def reset(self):
        """SGD no tiene estado interno."""
        pass

    def __repr__(self):
        return f"SGD(lr={self.lr})"


class MomentumSGD:
    """
    SGD con Momentum.

    Agrega un término de inercia (velocidad) que acumula gradientes
    pasados, permitiendo atravesar mínimos locales poco profundos:
        v_{t+1} = μ · v_t - η · ∇L
        W_{t+1} = W_t + v_{t+1}

    En la analogía física: es como una bola con masa que rueda por
    el paisaje de pérdida y puede superar pequeñas colinas.
    """

    name = 'Momentum'

    def __init__(self, lr=0.01, momentum=0.9):
        """
        Args:
            lr (float): Tasa de aprendizaje η.
            momentum (float): Coeficiente de inercia μ ∈ [0, 1).
        """
        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    def update(self, weights, gradients):
        """Actualiza los pesos con Momentum SGD."""
        if self.velocity is None:
            self.velocity = np.zeros_like(weights)
        self.velocity = self.momentum * self.velocity - self.lr * gradients
        return weights + self.velocity

    def reset(self):
        """Reinicia la velocidad acumulada."""
        self.velocity = None

    def __repr__(self):
        return f"MomentumSGD(lr={self.lr}, momentum={self.momentum})"


class Adam:
    """
    Algoritmo Adam (Adaptive Moment Estimation).

    Combina las ventajas de Momentum (primer momento) y RMSprop
    (segundo momento) con corrección de sesgo:
        m_t = β1·m_{t-1} + (1-β1)·g       (media de gradientes)
        v_t = β2·v_{t-1} + (1-β2)·g²      (varianza de gradientes)
        m̂_t = m_t / (1 - β1^t)            (corrección de sesgo)
        v̂_t = v_t / (1 - β2^t)
        W = W - η · m̂_t / (√v̂_t + ε)

    Es el optimizador más robusto y popular en deep learning.
    """

    name = 'Adam'

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Args:
            lr (float): Tasa de aprendizaje η.
            beta1 (float): Decaimiento del primer momento (media).
            beta2 (float): Decaimiento del segundo momento (varianza).
            epsilon (float): Constante numérica para evitar división por 0.
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None       # Primer momento (media)
        self.v = None       # Segundo momento (varianza)
        self.t = 0          # Contador de pasos

    def update(self, weights, gradients):
        """Actualiza los pesos con un paso de Adam."""
        if self.m is None:
            self.m = np.zeros_like(weights)
            self.v = np.zeros_like(weights)

        self.t += 1

        # Actualizar momentos
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradients ** 2

        # Corrección de sesgo
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        return weights - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def reset(self):
        """Reinicia los momentos acumulados."""
        self.m = None
        self.v = None
        self.t = 0

    def __repr__(self):
        return f"Adam(lr={self.lr}, β1={self.beta1}, β2={self.beta2})"


# ─── Factory ─────────────────────────────────────────────────────────

OPTIMIZERS = {
    'sgd': SGD,
    'momentum': MomentumSGD,
    'adam': Adam,
}


def create_optimizer(name, lr=None, **kwargs):
    """
    Crea un optimizador por nombre.

    Args:
        name (str): 'sgd', 'momentum', o 'adam'.
        lr (float): Tasa de aprendizaje (usa default si es None).
        **kwargs: Argumentos adicionales del optimizador.

    Returns:
        Instancia del optimizador.
    """
    cls = OPTIMIZERS.get(name.lower())
    if cls is None:
        available = ', '.join(OPTIMIZERS.keys())
        raise ValueError(
            f"Optimizador '{name}' no reconocido. Disponibles: {available}"
        )
    if lr is not None:
        kwargs['lr'] = lr
    return cls(**kwargs)


def list_optimizers():
    """Devuelve los nombres de optimizadores disponibles."""
    return list(OPTIMIZERS.keys())
