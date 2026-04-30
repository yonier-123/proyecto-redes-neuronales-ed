"""
trainer.py — Entrenador por descenso de gradiente.

Implementa el bucle de entrenamiento que ajusta los pesos θ de la red
neuronal para que la trayectoria generada se aproxime a una trayectoria
objetivo. El descenso de gradiente es en sí mismo una EDO:

    dW/dt = -η · ∇L(W)

discretizada como:  W_{t+1} = W_t - η · ∇L(W_t)

La función de pérdida es el error cuadrático medio entre
la trayectoria predicha y la objetivo:

    L = (1/N) Σ [ (x_pred - x_target)² + (y_pred - y_target)² ]

Los gradientes se calculan por diferencias finitas (numérico):
    ∂L/∂wᵢ ≈ [L(w + εeᵢ) - L(w)] / ε

Uso:
    >>> from core import NeuralField, ODESolver
    >>> from simulation.optimizers import SGD
    >>> field = NeuralField()
    >>> solver = ODESolver(field)
    >>> optimizer = SGD(lr=0.01)
    >>> trainer = GradientTrainer(field, solver, optimizer)
    >>> loss, grads = trainer.step(x0, y0, t_span, dt, xs_target, ys_target)
"""

import numpy as np


class GradientTrainer:
    """
    Entrena la red neuronal por descenso de gradiente numérico.

    Calcula gradientes por diferencias finitas y actualiza los pesos
    usando el optimizador proporcionado. Registra toda la historia
    del entrenamiento para visualización.

    Attributes:
        field (NeuralField): Red a entrenar.
        solver (ODESolver): Solver numérico para generar trayectorias.
        optimizer: Instancia del optimizador (SGD, Momentum, Adam).
        history (dict): Registro de loss, weights, gradients por época.
    """

    def __init__(self, neural_field, solver, optimizer, epsilon=1e-4):
        """
        Args:
            neural_field (NeuralField): Red neuronal a entrenar.
            solver (ODESolver): Solver de la EDO.
            optimizer: Optimizador (SGD, MomentumSGD, Adam).
            epsilon (float): Tamaño del paso para diferencias finitas.
        """
        self.field = neural_field
        self.solver = solver
        self.optimizer = optimizer
        self.epsilon = epsilon

        # Historia del entrenamiento (para gráficas de convergencia)
        self.history = {
            'loss': [],           # Pérdida en cada época
            'weights': [],        # Pesos en cada época
            'gradients': [],      # Gradientes en cada época
            'weight_deltas': [],  # Cambio en pesos ||ΔW||
        }

        self._epoch = 0

    def compute_loss(self, xs_pred, ys_pred, xs_target, ys_target):
        """
        Calcula el error cuadrático medio entre trayectorias.

        Args:
            xs_pred, ys_pred: Trayectoria generada por la red actual.
            xs_target, ys_target: Trayectoria objetivo.

        Returns:
            float: Pérdida L = mean((x_p - x_t)² + (y_p - y_t)²).
        """
        n = min(len(xs_pred), len(xs_target))
        loss = np.mean(
            (xs_pred[:n] - xs_target[:n]) ** 2 +
            (ys_pred[:n] - ys_target[:n]) ** 2
        )
        return float(loss)

    def compute_gradients(self, x0, y0, t_span, dt, xs_target, ys_target):
        """
        Calcula ∇L por diferencias finitas.

        Para cada peso wᵢ:
            ∂L/∂wᵢ ≈ [L(w + ε·eᵢ) - L(w)] / ε

        Args:
            x0, y0: Condición inicial.
            t_span: Intervalo de tiempo.
            dt: Paso de tiempo.
            xs_target, ys_target: Trayectoria objetivo.

        Returns:
            tuple: (gradients, base_loss) — gradiente y pérdida actual.
        """
        n_weights = self.field.NUM_WEIGHTS
        gradients = np.zeros(n_weights)
        w_current = self.field.get_weights()

        # Pérdida con pesos actuales
        base_loss = self._evaluate_loss(
            w_current, x0, y0, t_span, dt, xs_target, ys_target
        )

        # Gradiente por diferencias finitas (forward difference)
        for i in range(n_weights):
            w_perturbed = w_current.copy()
            w_perturbed[i] += self.epsilon
            loss_perturbed = self._evaluate_loss(
                w_perturbed, x0, y0, t_span, dt, xs_target, ys_target
            )
            gradients[i] = (loss_perturbed - base_loss) / self.epsilon

        # Restaurar pesos originales
        self.field.set_weights(w_current)

        return gradients, base_loss

    def _evaluate_loss(self, weights, x0, y0, t_span, dt,
                       xs_target, ys_target):
        """Evalúa la pérdida para un conjunto de pesos dado."""
        self.field.set_weights(weights)
        _, xs, ys = self.solver.solve_euler(x0, y0, t_span, dt)
        return self.compute_loss(xs, ys, xs_target, ys_target)

    def step(self, x0, y0, t_span, dt, xs_target, ys_target):
        """
        Ejecuta un paso de entrenamiento (una época).

        1. Calcula gradientes ∇L por diferencias finitas
        2. Actualiza pesos con el optimizador
        3. Registra todo en self.history

        Args:
            x0, y0: Condición inicial.
            t_span: Intervalo de tiempo.
            dt: Paso de tiempo.
            xs_target, ys_target: Trayectoria objetivo.

        Returns:
            tuple: (loss, gradients) — pérdida y gradientes de esta época.
        """
        # 1. Gradientes
        gradients, loss = self.compute_gradients(
            x0, y0, t_span, dt, xs_target, ys_target
        )

        # 2. Actualizar pesos
        w_old = self.field.get_weights()
        w_new = self.optimizer.update(w_old, gradients)
        self.field.set_weights(w_new)

        # 3. Registrar en historia
        delta_w = np.linalg.norm(w_new - w_old)
        self.history['loss'].append(loss)
        self.history['weights'].append(w_new.copy())
        self.history['gradients'].append(gradients.copy())
        self.history['weight_deltas'].append(delta_w)

        self._epoch += 1

        return loss, gradients

    def train(self, x0, y0, t_span, dt, xs_target, ys_target,
              max_epochs=500, tol=1e-6, verbose=True):
        """
        Ejecuta el bucle de entrenamiento completo.

        Args:
            x0, y0: Condición inicial.
            t_span: Intervalo de tiempo.
            dt: Paso de tiempo.
            xs_target, ys_target: Trayectoria objetivo.
            max_epochs (int): Máximo de épocas.
            tol (float): Tolerancia de pérdida para parar.
            verbose (bool): Imprimir progreso.

        Returns:
            dict: Historia completa del entrenamiento.
        """
        for epoch in range(max_epochs):
            loss, grads = self.step(
                x0, y0, t_span, dt, xs_target, ys_target
            )

            if verbose and epoch % 50 == 0:
                grad_norm = np.linalg.norm(grads)
                print(
                    f"  Época {epoch:4d} | "
                    f"Loss: {loss:.6f} | "
                    f"||∇L||: {grad_norm:.6f} | "
                    f"||ΔW||: {self.history['weight_deltas'][-1]:.6f}"
                )

            if loss < tol:
                if verbose:
                    print(f"  ✓ Convergencia alcanzada en época {epoch}")
                break

        return self.history

    def reset(self):
        """Reinicia la historia y el optimizador."""
        self.history = {
            'loss': [],
            'weights': [],
            'gradients': [],
            'weight_deltas': [],
        }
        self._epoch = 0
        self.optimizer.reset()

    @property
    def epoch(self):
        """Época actual."""
        return self._epoch

    @property
    def current_loss(self):
        """Última pérdida registrada, o None si no ha entrenado."""
        if self.history['loss']:
            return self.history['loss'][-1]
        return None
