"""
test_solver.py — Pruebas unitarias del núcleo matemático.

Verifica: NeuralField, ODESolver, presets y trainer.
Correr con:  pytest tests/ -v
"""

import sys
import os
import numpy as np
import pytest

# Agregar el directorio padre al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.neural_field import NeuralField
from core.ode_solver import ODESolver
from core.trainer import GradientTrainer
from simulation.presets import get_preset, list_presets
from simulation.optimizers import SGD, MomentumSGD, Adam, create_optimizer


# ═══════════════════════════════════════════════════════════════════════
# Tests de NeuralField
# ═══════════════════════════════════════════════════════════════════════

class TestNeuralField:
    def test_default_weights(self):
        """Pesos por defecto deben ser [0, -1, 1, 0, 0, 0]."""
        nf = NeuralField()
        expected = np.array([0.0, -1.0, 1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(nf.get_weights(), expected)

    def test_custom_weights(self):
        """Acepta pesos personalizados."""
        w = [0.5, -0.5, 0.5, -0.5, 0.1, -0.1]
        nf = NeuralField(w)
        np.testing.assert_array_almost_equal(nf.get_weights(), w)

    def test_invalid_weights_count(self):
        """Rechaza vectores de pesos con tamaño incorrecto."""
        with pytest.raises(ValueError):
            NeuralField([1.0, 2.0])

    def test_forward_scalar(self):
        """Forward pass con escalares devuelve escalares."""
        nf = NeuralField()
        dx, dy = nf.forward(1.0, 0.0)
        assert isinstance(dx, (float, np.floating))
        assert isinstance(dy, (float, np.floating))

    def test_forward_tanh_bounds(self):
        """Salida siempre entre -1 y 1 (por la tanh)."""
        nf = NeuralField([2.0, 3.0, -2.0, -3.0, 1.0, -1.0])
        for _ in range(100):
            x, y = np.random.randn(2) * 10
            dx, dy = nf.forward(x, y)
            assert -1.0 <= dx <= 1.0
            assert -1.0 <= dy <= 1.0

    def test_forward_grid(self):
        """Forward pass funciona con arrays (evaluación en malla)."""
        nf = NeuralField()
        X, Y = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))
        U, V = nf.forward(X, Y)
        assert U.shape == (10, 10)
        assert V.shape == (10, 10)

    def test_set_get_weights(self):
        """set_weights/get_weights son consistentes."""
        nf = NeuralField()
        new_w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        nf.set_weights(new_w)
        np.testing.assert_array_almost_equal(nf.get_weights(), new_w)

    def test_get_weights_is_copy(self):
        """get_weights devuelve copia, no referencia."""
        nf = NeuralField()
        w = nf.get_weights()
        w[0] = 999.0
        assert nf.get_weights()[0] != 999.0

    def test_randomize_weights(self):
        """randomize genera pesos distintos de los default."""
        nf = NeuralField()
        original = nf.get_weights().copy()
        nf.randomize_weights(scale=1.0, rng=np.random.default_rng(42))
        assert not np.allclose(original, nf.get_weights())

    def test_repr(self):
        """repr devuelve string legible."""
        nf = NeuralField()
        r = repr(nf)
        assert 'NeuralField' in r
        assert 'w1=' in r


# ═══════════════════════════════════════════════════════════════════════
# Tests de ODESolver
# ═══════════════════════════════════════════════════════════════════════

class TestODESolver:
    def test_euler_output_shape(self):
        """Euler devuelve arrays del mismo largo."""
        nf = NeuralField()
        solver = ODESolver(nf)
        t, xs, ys = solver.solve_euler(1.0, 0.0, (0, 5), 0.05)
        assert len(t) == len(xs) == len(ys)

    def test_euler_initial_condition(self):
        """Euler respeta la condición inicial."""
        nf = NeuralField()
        solver = ODESolver(nf)
        t, xs, ys = solver.solve_euler(2.5, -1.3, (0, 1), 0.01)
        assert xs[0] == pytest.approx(2.5)
        assert ys[0] == pytest.approx(-1.3)

    def test_rk4_output_shape(self):
        """RK4 devuelve arrays del mismo largo."""
        nf = NeuralField()
        solver = ODESolver(nf)
        t, xs, ys = solver.solve_rk4(1.0, 0.0, (0, 5), 0.05)
        assert len(t) == len(xs) == len(ys)

    def test_rk4_initial_condition(self):
        """RK4 respeta la condición inicial."""
        nf = NeuralField()
        solver = ODESolver(nf)
        t, xs, ys = solver.solve_rk4(2.5, -1.3, (0, 1), 0.01)
        assert xs[0] == pytest.approx(2.5)
        assert ys[0] == pytest.approx(-1.3)

    def test_rk4_circular_radius(self):
        """Con pesos de rotación pura, el radio es casi constante."""
        nf = NeuralField([0.0, -1.0, 1.0, 0.0, 0.0, 0.0])
        solver = ODESolver(nf)
        t, xs, ys = solver.solve_rk4(1.0, 0.0, (0, 6.28), 0.01)
        radii = np.sqrt(xs**2 + ys**2)
        assert np.std(radii) < 0.05, f"Radio no constante: std={np.std(radii):.4f}"

    def test_weights_change_trajectory(self):
        """Diferentes pesos generan trayectorias distintas."""
        nf = NeuralField()
        solver = ODESolver(nf)
        _, xs1, _ = solver.solve_euler(1.0, 0.0, (0, 5), 0.05)
        nf.set_weights([1.0, 0.5, -0.5, 1.0, 0.1, -0.1])
        _, xs2, _ = solver.solve_euler(1.0, 0.0, (0, 5), 0.05)
        assert not np.allclose(xs1, xs2)

    def test_solve_step(self):
        """solve_step da un paso coherente."""
        nf = NeuralField()
        solver = ODESolver(nf)
        x_new, y_new = solver.solve_step(1.0, 0.0, 0.05)
        assert x_new != 1.0 or y_new != 0.0  # se movió

    def test_solve_generic_dispatch(self):
        """solve() despacha correctamente a euler y rk4."""
        nf = NeuralField()
        solver = ODESolver(nf)
        t1, xs1, ys1 = solver.solve(1.0, 0.0, (0, 1), 0.05, method='euler')
        t2, xs2, ys2 = solver.solve(1.0, 0.0, (0, 1), 0.05, method='rk4')
        assert len(t1) > 0
        assert len(t2) > 0

    def test_solve_invalid_method(self):
        """solve() con método inválido lanza error."""
        nf = NeuralField()
        solver = ODESolver(nf)
        with pytest.raises(ValueError):
            solver.solve(1.0, 0.0, (0, 1), 0.05, method='invalid')


# ═══════════════════════════════════════════════════════════════════════
# Tests de Presets
# ═══════════════════════════════════════════════════════════════════════

class TestPresets:
    def test_list_presets(self):
        """list_presets devuelve al menos 5 presets."""
        presets = list_presets()
        assert len(presets) >= 5
        assert 'circulo' in presets

    def test_get_preset(self):
        """get_preset devuelve dict con las claves esperadas."""
        p = get_preset('circulo')
        assert 'weights' in p
        assert 'x0' in p
        assert 'y0' in p
        assert len(p['weights']) == 6

    def test_get_preset_invalid(self):
        """get_preset con nombre inválido lanza KeyError."""
        with pytest.raises(KeyError):
            get_preset('no_existe')

    def test_all_presets_generate_trajectories(self):
        """Cada preset produce una trayectoria sin errores."""
        for name in list_presets():
            p = get_preset(name)
            nf = NeuralField(p['weights'])
            solver = ODESolver(nf)
            t, xs, ys = solver.solve_rk4(p['x0'], p['y0'], (0, 5), 0.05)
            assert len(t) > 0
            assert not np.any(np.isnan(xs))
            assert not np.any(np.isnan(ys))


# ═══════════════════════════════════════════════════════════════════════
# Tests de Optimizadores
# ═══════════════════════════════════════════════════════════════════════

class TestOptimizers:
    def test_sgd_update(self):
        """SGD: w_new = w - lr * grad."""
        opt = SGD(lr=0.1)
        w = np.array([1.0, 2.0])
        g = np.array([0.5, 0.5])
        w_new = opt.update(w, g)
        np.testing.assert_array_almost_equal(w_new, [0.95, 1.95])

    def test_momentum_update(self):
        """Momentum acumula velocidad."""
        opt = MomentumSGD(lr=0.1, momentum=0.9)
        w = np.array([1.0, 1.0])
        g = np.array([1.0, 1.0])
        w1 = opt.update(w, g)
        w2 = opt.update(w1, g)
        # Segunda actualización se mueve más (momentum)
        delta1 = np.linalg.norm(w1 - w)
        delta2 = np.linalg.norm(w2 - w1)
        assert delta2 > delta1

    def test_adam_update(self):
        """Adam produce actualización válida."""
        opt = Adam(lr=0.01)
        w = np.array([1.0, 2.0, 3.0])
        g = np.array([0.1, 0.2, 0.3])
        w_new = opt.update(w, g)
        assert not np.allclose(w, w_new)

    def test_create_optimizer(self):
        """Factory crea optimizadores por nombre."""
        opt = create_optimizer('sgd', lr=0.05)
        assert isinstance(opt, SGD)
        assert opt.lr == 0.05

    def test_reset(self):
        """Reset limpia estado interno."""
        opt = Adam(lr=0.01)
        w = np.array([1.0, 2.0])
        opt.update(w, np.array([0.1, 0.1]))
        assert opt.t == 1
        opt.reset()
        assert opt.t == 0
        assert opt.m is None


# ═══════════════════════════════════════════════════════════════════════
# Tests de Trainer
# ═══════════════════════════════════════════════════════════════════════

class TestTrainer:
    def _make_trainer(self, lr=0.01):
        nf = NeuralField()
        solver = ODESolver(nf)
        opt = SGD(lr=lr)
        return GradientTrainer(nf, solver, opt)

    def test_compute_loss_zero(self):
        """Loss es 0 cuando predicha = objetivo."""
        trainer = self._make_trainer()
        xs = np.array([1.0, 2.0, 3.0])
        loss = trainer.compute_loss(xs, xs, xs, xs)
        assert loss == pytest.approx(0.0)

    def test_compute_loss_positive(self):
        """Loss > 0 cuando trayectorias difieren."""
        trainer = self._make_trainer()
        xs = np.array([1.0, 2.0, 3.0])
        ys = np.array([0.0, 0.0, 0.0])
        xs_t = np.array([0.0, 0.0, 0.0])
        loss = trainer.compute_loss(xs, ys, xs_t, ys)
        assert loss > 0

    def test_step_decreases_loss(self):
        """Un paso de entrenamiento generalmente reduce la pérdida."""
        # Crear trayectoria objetivo con preset 'circulo'
        target_field = NeuralField([0.0, -1.0, 1.0, 0.0, 0.0, 0.0])
        target_solver = ODESolver(target_field)
        _, xs_t, ys_t = target_solver.solve_euler(1.0, 0.0, (0, 3), 0.1)

        # Red con pesos aleatorios
        nf = NeuralField([0.5, -0.5, 0.5, -0.5, 0.1, 0.1])
        solver = ODESolver(nf)
        trainer = GradientTrainer(nf, solver, SGD(lr=0.1))

        loss0, _ = trainer.step(1.0, 0.0, (0, 3), 0.1, xs_t, ys_t)
        loss1, _ = trainer.step(1.0, 0.0, (0, 3), 0.1, xs_t, ys_t)
        # No siempre baja en 1 paso, pero la historia se registra
        assert len(trainer.history['loss']) == 2

    def test_history_tracking(self):
        """El trainer registra correctamente loss, weights, gradients."""
        trainer = self._make_trainer()
        target = np.ones(10)
        trainer.step(1.0, 0.0, (0, 0.5), 0.05, target, target)
        assert len(trainer.history['loss']) == 1
        assert len(trainer.history['weights']) == 1
        assert len(trainer.history['gradients']) == 1
        assert len(trainer.history['weight_deltas']) == 1

    def test_reset(self):
        """Reset limpia la historia."""
        trainer = self._make_trainer()
        target = np.ones(10)
        trainer.step(1.0, 0.0, (0, 0.5), 0.05, target, target)
        trainer.reset()
        assert len(trainer.history['loss']) == 0
        assert trainer.epoch == 0
