"""
main.py — Punto de entrada del Neural ODE Simulator.

Modos de ejecución:
    python main.py                      → Aplicación interactiva (MVP)
    python main.py --mode app           → Aplicación interactiva (MVP)
    python main.py --mode all           → Demo estática de todos los presets
    python main.py --mode static        → Un preset con campo vectorial
    python main.py --mode train         → Demo de entrenamiento completo
    python main.py --mode activations   → Heatmaps de activaciones

Ejemplo:
    python main.py --mode train --preset circulo --optimizer adam --lr 0.05 --epochs 200
"""

import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from core.neural_field import NeuralField
from core.ode_solver import ODESolver
from core.trainer import GradientTrainer
from simulation.presets import get_preset, list_presets, PRESETS
from simulation.optimizers import create_optimizer
from visualization.static_plot import (
    plot_trajectory_with_field,
    plot_comparison,
    plot_activations,
    plot_all_presets,
    plot_training_history,
)
import config


def demo_static(preset_name='circulo'):
    """Muestra la trayectoria de un preset con campo vectorial."""
    print(f"\n{'='*60}")
    print(f"  DEMO ESTÁTICA — Preset: {preset_name}")
    print(f"{'='*60}")

    preset = get_preset(preset_name)
    print(f"  Descripción: {preset['descripcion']}")
    print(f"  Pesos: {preset['weights']}")
    print(f"  Condición inicial: ({preset['x0']}, {preset['y0']})")

    field = NeuralField(preset['weights'])
    solver = ODESolver(field)
    t, xs, ys = solver.solve_rk4(
        preset['x0'], preset['y0'],
        (config.T_START, config.T_END), config.DT
    )

    print(f"  Puntos generados: {len(t)}")
    print(f"  Rango X: [{xs.min():.2f}, {xs.max():.2f}]")
    print(f"  Rango Y: [{ys.min():.2f}, {ys.max():.2f}]")

    fig = plot_trajectory_with_field(field, xs, ys)
    plt.show()


def demo_all_presets():
    """Muestra panel con todos los presets disponibles."""
    print(f"\n{'='*60}")
    print(f"  PANEL DE PRESETS — {len(PRESETS)} configuraciones")
    print(f"{'='*60}")

    for name in list_presets():
        p = get_preset(name)
        print(f"  • {name:20s} → {p['descripcion']}")

    fig = plot_all_presets(PRESETS, ODESolver, NeuralField,
                          t_span=(config.T_START, config.T_END), dt=config.DT)
    plt.show()


def demo_train(preset_name='circulo', optimizer_name='sgd',
               lr=None, max_epochs=200):
    """Demo de entrenamiento: aprende a aproximar una trayectoria objetivo."""
    print(f"\n{'='*60}")
    print(f"  DEMO DE ENTRENAMIENTO")
    print(f"  Objetivo: {preset_name}  |  Optimizador: {optimizer_name}")
    print(f"{'='*60}")

    # 1. Generar trayectoria objetivo
    preset = get_preset(preset_name)
    target_field = NeuralField(preset['weights'])
    target_solver = ODESolver(target_field)
    t_span = (config.T_START, min(config.T_END, 5.0))  # Recortar para velocidad
    _, xs_target, ys_target = target_solver.solve_euler(
        preset['x0'], preset['y0'], t_span, config.DT
    )
    print(f"  Trayectoria objetivo: {len(xs_target)} puntos")
    print(f"  Pesos objetivo: {preset['weights']}")

    # 2. Crear red con pesos aleatorios
    field = NeuralField()
    field.randomize_weights(scale=0.5, rng=np.random.default_rng(42))
    solver = ODESolver(field)
    print(f"  Pesos iniciales: {[f'{w:.3f}' for w in field.get_weights()]}")

    # 3. Crear optimizador y trainer
    if lr is None:
        lr = {'sgd': 0.1, 'momentum': 0.05, 'adam': 0.05}.get(optimizer_name, 0.05)

    optimizer = create_optimizer(optimizer_name, lr=lr)
    trainer = GradientTrainer(field, solver, optimizer)

    print(f"  Learning rate: {lr}")
    print(f"  Épocas máximas: {max_epochs}")
    print(f"\n  Entrenando...")

    # 4. Entrenar
    history = trainer.train(
        preset['x0'], preset['y0'], t_span, config.DT,
        xs_target, ys_target,
        max_epochs=max_epochs, tol=1e-5, verbose=True
    )

    # 5. Resultado final
    print(f"\n  Pesos finales: {[f'{w:.3f}' for w in field.get_weights()]}")
    print(f"  Loss final: {history['loss'][-1]:.6f}")

    # 6. Generar trayectoria final y comparar
    _, xs_pred, ys_pred = solver.solve_euler(
        preset['x0'], preset['y0'], t_span, config.DT
    )

    fig1 = plot_comparison(xs_pred, ys_pred, xs_target, ys_target,
                           title=f'Resultado — {optimizer_name.upper()} (lr={lr})')
    fig2 = plot_training_history(history)
    plt.show()


def demo_activations(preset_name='circulo'):
    """Muestra los heatmaps de activaciones."""
    preset = get_preset(preset_name)
    field = NeuralField(preset['weights'])
    fig = plot_activations(field)
    plt.show()


def demo_app():
    """Lanza la aplicación interactiva PyQt5 (MVP)."""
    from PyQt5.QtWidgets import QApplication
    from ui.controls import NeuralODEApp

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = NeuralODEApp()
    window.show()
    sys.exit(app.exec_())


def main():
    parser = argparse.ArgumentParser(
        description='Neural ODE Simulator — Ecuaciones Diferenciales',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py                                  Aplicación interactiva (MVP)
  python main.py --mode all                        Panel de todos los presets
  python main.py --mode static --preset circulo    Un preset con campo vectorial
  python main.py --mode train --optimizer adam      Entrenar con Adam
  python main.py --mode activations                Heatmaps de activaciones
        """
    )
    parser.add_argument('--mode', default='app',
                        choices=['app', 'all', 'static', 'train', 'activations'],
                        help='Modo de ejecución (default: app)')
    parser.add_argument('--preset', default='circulo',
                        choices=list_presets(),
                        help='Preset de trayectoria')
    parser.add_argument('--optimizer', default='sgd',
                        choices=['sgd', 'momentum', 'adam'],
                        help='Optimizador para entrenamiento')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default según optimizador)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Épocas de entrenamiento')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("  NEURAL ODE SIMULATOR")
    print("  Ecuaciones Diferenciales + Redes Neuronales")
    print("="*60)

    if args.mode == 'app':
        demo_app()
    elif args.mode == 'all':
        matplotlib.use('TkAgg')
        demo_all_presets()
    elif args.mode == 'static':
        matplotlib.use('TkAgg')
        demo_static(args.preset)
    elif args.mode == 'train':
        matplotlib.use('TkAgg')
        demo_train(args.preset, args.optimizer, args.lr, args.epochs)
    elif args.mode == 'activations':
        matplotlib.use('TkAgg')
        demo_activations(args.preset)


if __name__ == '__main__':
    main()
