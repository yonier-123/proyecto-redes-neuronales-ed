"""
static_plot.py — Visualizaciones estáticas del sistema dinámico.

Genera gráficas de trayectorias 2D con campo vectorial, comparaciones,
heatmaps de activaciones y métricas de entrenamiento.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def _apply_dark_style():
    """Aplica estilo oscuro moderno."""
    plt.rcParams.update({
        'figure.facecolor': '#0f0f1a',
        'axes.facecolor': '#16162a',
        'axes.edgecolor': '#2a2a4a',
        'axes.labelcolor': '#c0c0d0',
        'axes.grid': True,
        'grid.color': '#1e1e3a',
        'grid.alpha': 0.5,
        'text.color': '#e0e0f0',
        'xtick.color': '#8080a0',
        'ytick.color': '#8080a0',
        'font.size': 11,
    })


def _gradient_line(xs, ys, cmap='cool', lw=2.5):
    """Línea con color degradado temporal."""
    pts = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(0, len(xs)), linewidth=lw)
    lc.set_array(np.arange(len(xs)))
    return lc


def plot_trajectory_with_field(neural_field, xs, ys,
                               plot_range=(-3, 3), grid_size=20,
                               title=None, figsize=(10, 10)):
    """Trayectoria 2D sobre campo vectorial (streamplot)."""
    _apply_dark_style()
    fig, ax = plt.subplots(figsize=figsize)

    xr = np.linspace(*plot_range, grid_size)
    yr = np.linspace(*plot_range, grid_size)
    X, Y = np.meshgrid(xr, yr)
    U, V = neural_field.get_field(X, Y)
    speed = np.sqrt(U**2 + V**2)

    ax.streamplot(X, Y, U, V, color=speed, cmap='viridis',
                  density=1.2, linewidth=0.8, arrowsize=1.0)
    ax.add_collection(_gradient_line(xs, ys))
    ax.plot(xs[0], ys[0], '*', color='#00ff88', ms=15, zorder=5,
            label=f'Inicio ({xs[0]:.1f}, {ys[0]:.1f})')
    ax.plot(xs[-1], ys[-1], 'o', color='#ff4444', ms=10, zorder=5,
            label=f'Final ({xs[-1]:.2f}, {ys[-1]:.2f})')

    ax.set_xlim(plot_range); ax.set_ylim(plot_range)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=10, facecolor='#1a1a2e',
              edgecolor='#3a3a5a', labelcolor='#c0c0d0')

    if title:
        ax.set_title(title, fontsize=15, fontweight='bold', color='#e0e0f0')
    else:
        w = neural_field.get_weights_dict()
        ax.set_title('Trayectoria + Campo Vectorial\n' +
                      '  '.join(f'{k}={v:+.2f}' for k, v in w.items()),
                      fontsize=13, color='#a0a0c0')
    fig.tight_layout()
    return fig


def plot_comparison(xs_pred, ys_pred, xs_target, ys_target,
                    title='Predicha vs Objetivo', figsize=(10, 10)):
    """Compara trayectoria predicha y objetivo."""
    _apply_dark_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(xs_target, ys_target, '--', color='#ff6b6b', lw=2, alpha=0.7, label='Objetivo')
    ax.add_collection(_gradient_line(xs_pred, ys_pred))
    ax.plot(xs_pred[0], ys_pred[0], '*', color='#00ff88', ms=15, zorder=5, label='Inicio')
    ax.plot([], [], '-', color='#4fc3f7', label='Predicha')
    m = 0.5
    ax.set_xlim(min(xs_pred.min(), xs_target.min())-m, max(xs_pred.max(), xs_target.max())+m)
    ax.set_ylim(min(ys_pred.min(), ys_target.min())-m, max(ys_pred.max(), ys_target.max())+m)
    ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(title, fontsize=15, fontweight='bold', color='#e0e0f0')
    ax.legend(fontsize=10, facecolor='#1a1a2e', edgecolor='#3a3a5a', labelcolor='#c0c0d0')
    fig.tight_layout()
    return fig


def plot_activations(neural_field, plot_range=(-3, 3), resolution=100, figsize=(16, 7)):
    """Heatmaps de activaciones f_θ y g_θ."""
    _apply_dark_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    xr = np.linspace(*plot_range, resolution)
    yr = np.linspace(*plot_range, resolution)
    X, Y = np.meshgrid(xr, yr)
    U, V = neural_field.get_field(X, Y)
    ext = [plot_range[0], plot_range[1], plot_range[0], plot_range[1]]

    for ax, data, lbl in [(ax1, U, 'f_θ(x,y) = dx/dt'), (ax2, V, 'g_θ(x,y) = dy/dt')]:
        im = ax.imshow(data, extent=ext, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        ax.set_title(lbl, fontsize=13, color='#e0e0f0')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, shrink=0.8, label='Activación')

    w = neural_field.get_weights_dict()
    fig.suptitle('Heatmaps de Activaciones — ' + '  '.join(f'{k}={v:+.2f}' for k, v in w.items()),
                 fontsize=14, fontweight='bold', color='#e0e0f0', y=1.02)
    fig.tight_layout()
    return fig


def plot_all_presets(presets_dict, solver_class, field_class,
                    t_span=(0, 10), dt=0.01, figsize=(18, 12)):
    """Panel con trayectoria de cada preset."""
    _apply_dark_style()
    n = len(presets_dict)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for idx, (name, preset) in enumerate(presets_dict.items()):
        ax = axes[idx]
        field = field_class(preset['weights'])
        solver = solver_class(field)
        t, xs, ys = solver.solve_rk4(preset['x0'], preset['y0'], t_span, dt)
        ax.add_collection(_gradient_line(xs, ys, lw=2.0))
        ax.plot(xs[0], ys[0], '*', color='#00ff88', ms=12, zorder=5)
        ax.plot(xs[-1], ys[-1], 'o', color='#ff4444', ms=8, zorder=5)
        m = 0.3
        ax.set_xlim(xs.min()-m, xs.max()+m)
        ax.set_ylim(ys.min()-m, ys.max()+m)
        ax.set_aspect('equal')
        ax.set_title(f"{name}\n{preset.get('descripcion','')}",
                     fontsize=10, color=preset.get('color', '#e0e0f0'))

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Trayectorias por Preset', fontsize=16, fontweight='bold', color='#e0e0f0')
    fig.tight_layout()
    return fig


def plot_training_history(history, figsize=(16, 10)):
    """4 subplots: pérdida, ||∇L||, pesos, ||ΔW|| vs época."""
    _apply_dark_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    epochs = np.arange(len(history['loss']))
    w_names = ['w1','w2','w3','w4','b1','b2']
    w_colors = ['#4fc3f7','#00ff88','#ff6b6b','#e040fb','#ffab40','#7c4dff']

    axes[0,0].plot(epochs, history['loss'], color='#ff6b6b', lw=2)
    axes[0,0].set_title('Pérdida (Loss)', color='#e0e0f0')
    axes[0,0].set_xlabel('Época'); axes[0,0].set_ylabel('L'); axes[0,0].set_yscale('log')

    grad_n = [np.linalg.norm(g) for g in history['gradients']]
    axes[0,1].plot(epochs, grad_n, color='#ffab40', lw=2)
    axes[0,1].set_title('||∇L||', color='#e0e0f0')
    axes[0,1].set_xlabel('Época'); axes[0,1].set_ylabel('||∇L||'); axes[0,1].set_yscale('log')

    wa = np.array(history['weights'])
    for i, (nm, c) in enumerate(zip(w_names, w_colors)):
        axes[1,0].plot(epochs, wa[:, i], color=c, lw=1.5, label=nm)
    axes[1,0].set_title('Evolución de pesos', color='#e0e0f0')
    axes[1,0].set_xlabel('Época'); axes[1,0].set_ylabel('Valor')
    axes[1,0].legend(fontsize=8, facecolor='#1a1a2e', edgecolor='#3a3a5a',
                     labelcolor='#c0c0d0', ncol=3)

    axes[1,1].plot(epochs, history['weight_deltas'], color='#7c4dff', lw=2)
    axes[1,1].set_title('||ΔW||', color='#e0e0f0')
    axes[1,1].set_xlabel('Época'); axes[1,1].set_ylabel('||ΔW||'); axes[1,1].set_yscale('log')

    fig.suptitle('Métricas de Entrenamiento', fontsize=16, fontweight='bold', color='#e0e0f0')
    fig.tight_layout()
    return fig
