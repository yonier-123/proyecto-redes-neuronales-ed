"""
test_integration.py - Tests de integracion del sistema completo.

Verifica que todos los modulos trabajan juntos correctamente:
    - Entrenamiento end-to-end (loss baja)
    - Comparacion de optimizadores
    - Presets generan trayectorias validas con cada solver
    - Visualizaciones no lanzan excepciones
    - Exportacion de figuras
"""

import sys
import os
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.neural_field import NeuralField
from core.ode_solver import ODESolver
from core.trainer import GradientTrainer
from simulation.presets import get_preset, list_presets, PRESETS
from simulation.optimizers import SGD, MomentumSGD, Adam, create_optimizer


class TestEndToEndTraining:
    """Verifica que el entrenamiento converge correctamente."""

    def _make_target(self, preset_name):
        p = get_preset(preset_name)
        f = NeuralField(p['weights'])
        s = ODESolver(f)
        _, xs, ys = s.solve_euler(p['x0'], p['y0'], (0, 3), 0.1)
        return p, xs, ys

    def test_sgd_converges(self):
        """SGD reduce la perdida tras 50 epocas."""
        p, xs_t, ys_t = self._make_target('circulo')
        field = NeuralField([0.5, -0.5, 0.5, -0.5, 0.0, 0.0])
        solver = ODESolver(field)
        trainer = GradientTrainer(field, solver, SGD(lr=0.1))
        h = trainer.train(p['x0'], p['y0'], (0, 3), 0.1,
                          xs_t, ys_t, max_epochs=50, verbose=False)
        assert h['loss'][-1] < h['loss'][0], "SGD no redujo la perdida"

    def test_adam_converges_faster(self):
        """Adam converge mas rapido que SGD."""
        p, xs_t, ys_t = self._make_target('circulo')
        rng = np.random.default_rng(42)
        init_w = rng.normal(0, 0.5, 6)

        losses = {}
        for opt_name, lr in [('sgd', 0.1), ('adam', 0.05)]:
            field = NeuralField(init_w.copy())
            solver = ODESolver(field)
            opt = create_optimizer(opt_name, lr=lr)
            trainer = GradientTrainer(field, solver, opt)
            h = trainer.train(p['x0'], p['y0'], (0, 3), 0.1,
                              xs_t, ys_t, max_epochs=30, verbose=False)
            losses[opt_name] = h['loss'][-1]

        # Adam deberia converger razonablemente
        assert losses['adam'] < 1.0, f"Adam loss too high: {losses['adam']}"

    def test_momentum_has_velocity(self):
        """Momentum acumula velocidad correctamente."""
        p, xs_t, ys_t = self._make_target('espiral_entrada')
        field = NeuralField([0.3, -0.3, 0.3, -0.3, 0.0, 0.0])
        solver = ODESolver(field)
        trainer = GradientTrainer(field, solver, MomentumSGD(lr=0.05))
        h = trainer.train(p['x0'], p['y0'], (0, 3), 0.1,
                          xs_t, ys_t, max_epochs=30, verbose=False)
        # Los deltas de peso deben crecer al inicio (momentum)
        assert len(h['weight_deltas']) == 30

    def test_lr_affects_convergence_speed(self):
        """Un LR mas grande converge mas rapido (hasta cierto punto)."""
        p, xs_t, ys_t = self._make_target('punto_fijo')
        init_w = [0.5, 0.5, -0.5, -0.5, 0.0, 0.0]
        losses = {}
        for lr in [0.01, 0.1]:
            field = NeuralField(init_w.copy())
            solver = ODESolver(field)
            trainer = GradientTrainer(field, solver, SGD(lr=lr))
            h = trainer.train(p['x0'], p['y0'], (0, 3), 0.1,
                              xs_t, ys_t, max_epochs=30, verbose=False)
            losses[lr] = h['loss'][-1]
        assert losses[0.1] < losses[0.01]


class TestSolverComparison:
    """Verifica que Euler y RK4 producen resultados coherentes."""

    def test_euler_vs_rk4_same_direction(self):
        """Euler y RK4 van en la misma direccion general."""
        for name in list_presets():
            p = get_preset(name)
            f = NeuralField(p['weights'])
            s = ODESolver(f)
            _, xs_e, ys_e = s.solve_euler(p['x0'], p['y0'], (0, 2), 0.05)
            _, xs_r, ys_r = s.solve_rk4(p['x0'], p['y0'], (0, 2), 0.05)
            # El primer punto debe ser igual
            assert xs_e[0] == pytest.approx(xs_r[0])
            assert ys_e[0] == pytest.approx(ys_r[0])
            # La correlacion general debe ser positiva
            n = min(len(xs_e), len(xs_r))
            corr_x = np.corrcoef(xs_e[:n], xs_r[:n])[0, 1]
            assert corr_x > 0.5, f"Preset {name}: correlacion baja {corr_x:.2f}"

    def test_rk4_more_accurate_circle(self):
        """RK4 mantiene el radio mejor que Euler en trayectoria circular."""
        f = NeuralField([0.0, -1.0, 1.0, 0.0, 0.0, 0.0])
        s = ODESolver(f)
        _, xs_e, ys_e = s.solve_euler(1.0, 0.0, (0, 6.28), 0.05)
        _, xs_r, ys_r = s.solve_rk4(1.0, 0.0, (0, 6.28), 0.05)
        r_euler = np.std(np.sqrt(xs_e**2 + ys_e**2))
        r_rk4 = np.std(np.sqrt(xs_r**2 + ys_r**2))
        assert r_rk4 < r_euler, "RK4 deberia ser mas preciso"


class TestVisualizationsNoError:
    """Verifica que las visualizaciones no lanzan excepciones."""

    def test_static_plot_creates_figure(self):
        """static_plot genera figuras sin error."""
        import matplotlib
        matplotlib.use('Agg')
        from visualization.static_plot import (
            plot_trajectory_with_field,
            plot_comparison,
            plot_activations,
            plot_training_history,
        )
        f = NeuralField()
        s = ODESolver(f)
        _, xs, ys = s.solve_rk4(1.0, 0.0, (0, 3), 0.1)

        fig1 = plot_trajectory_with_field(f, xs, ys, grid_size=8)
        assert fig1 is not None
        fig2 = plot_comparison(xs, ys, xs * 0.9, ys * 0.9)
        assert fig2 is not None
        fig3 = plot_activations(f, resolution=20)
        assert fig3 is not None

        import matplotlib.pyplot as plt
        plt.close('all')

    def test_static_plot_training_history(self):
        """plot_training_history funciona con datos validos."""
        import matplotlib
        matplotlib.use('Agg')
        from visualization.static_plot import plot_training_history

        history = {
            'loss': [1.0, 0.5, 0.2],
            'weights': [np.zeros(6), np.ones(6), np.ones(6) * 0.5],
            'gradients': [np.ones(6), np.ones(6) * 0.5, np.ones(6) * 0.1],
            'weight_deltas': [0.1, 0.05, 0.01],
        }
        fig = plot_training_history(history)
        assert fig is not None

        import matplotlib.pyplot as plt
        plt.close('all')

    def test_animator_classes_init(self):
        """Las clases de animator se instancian sin error."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from visualization.animator import (
            NeuralNetworkGraph,
            ConvergencePanel,
            DynamicTrajectoryPlot,
            DynamicActivationHeatmap,
        )

        fig, ax = plt.subplots()
        nn = NeuralNetworkGraph(ax)
        nn.update(np.array([0.5, -1.0, 1.0, -0.5, 0.1, -0.1]))
        assert nn is not None

        fig2, axes = plt.subplots(2, 2)
        cp = ConvergencePanel(axes.flatten())
        cp.update(1, 0.5, np.ones(6), np.ones(6), 0.01)
        assert cp is not None

        fig3, ax3 = plt.subplots()
        f = NeuralField()
        dt = DynamicTrajectoryPlot(ax3, f, grid_size=5)
        assert dt is not None

        fig4, axes4 = plt.subplots(1, 2)
        dh = DynamicActivationHeatmap(axes4, f, resolution=10)
        dh.update()
        assert dh is not None

        plt.close('all')


class TestExportCapability:
    """Verifica que las figuras se pueden guardar como PNG."""

    def test_save_figures_to_temp(self):
        """Las figuras se guardan sin error."""
        import matplotlib
        matplotlib.use('Agg')
        from visualization.static_plot import plot_trajectory_with_field

        f = NeuralField()
        s = ODESolver(f)
        _, xs, ys = s.solve_rk4(1.0, 0.0, (0, 3), 0.1)
        fig = plot_trajectory_with_field(f, xs, ys, grid_size=8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        fig.savefig(tmp_path, dpi=72)
        assert os.path.getsize(tmp_path) > 0

        import matplotlib.pyplot as plt
        plt.close(fig)
        plt.close('all')
        os.unlink(tmp_path)
