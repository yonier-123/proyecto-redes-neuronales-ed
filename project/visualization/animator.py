"""
animator.py — Visualizaciones animadas del sistema.

Contiene:
    - LossLandscape3D: Paisaje de pérdida 3D con bola que rueda
    - NeuralNetworkGraph: Grafo animado de la red neuronal
    - VectorFieldPlot: Campo vectorial dinámico con trayectoria
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, colors
from mpl_toolkits.mplot3d import Axes3D


# ═══════════════════════════════════════════════════════════════════════
# Paisaje de Pérdida 3D
# ═══════════════════════════════════════════════════════════════════════

class LossLandscape3D:
    """
    Superficie 3D del paisaje de pérdida con bola animada.

    Eje X = w1, Eje Y = w2, Eje Z = Loss.
    La bola roja se mueve siguiendo el camino del optimizador.
    """

    def __init__(self, ax, field, solver, x0, y0, t_span, dt,
                 xs_target, ys_target, w_range=(-2, 2), resolution=30):
        self.ax = ax
        self.field = field
        self.solver = solver
        self.x0, self.y0 = x0, y0
        self.t_span, self.dt = t_span, dt
        self.xs_target = xs_target
        self.ys_target = ys_target

        # Calcular superficie de pérdida (fijando w3,w4,b1,b2)
        w = field.get_weights()
        w1_range = np.linspace(w_range[0], w_range[1], resolution)
        w2_range = np.linspace(w_range[0], w_range[1], resolution)
        self.W1, self.W2 = np.meshgrid(w1_range, w2_range)
        self.Z = np.zeros_like(self.W1)

        for i in range(resolution):
            for j in range(resolution):
                test_w = w.copy()
                test_w[0] = self.W1[i, j]
                test_w[1] = self.W2[i, j]
                field.set_weights(test_w)
                _, xs, ys = solver.solve_euler(x0, y0, t_span, dt)
                n = min(len(xs), len(xs_target))
                self.Z[i, j] = np.mean(
                    (xs[:n] - xs_target[:n])**2 +
                    (ys[:n] - ys_target[:n])**2
                )
        field.set_weights(w)  # Restaurar

        # Trail del optimizador
        self.trail_w1 = []
        self.trail_w2 = []
        self.trail_loss = []

        self._draw_surface()

    def _draw_surface(self):
        self.ax.set_facecolor('#0f0f1a')
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False

        # Limitar Z para mejor visualización
        z_max = np.percentile(self.Z, 95)
        Z_clip = np.clip(self.Z, 0, z_max)

        self.surface = self.ax.plot_surface(
            self.W1, self.W2, Z_clip,
            cmap='viridis', alpha=0.6, edgecolor='none'
        )
        self.ax.set_xlabel('w1', color='#c0c0d0', fontsize=10)
        self.ax.set_ylabel('w2', color='#c0c0d0', fontsize=10)
        self.ax.set_zlabel('Loss', color='#c0c0d0', fontsize=10)
        self.ax.set_title('Paisaje de Pérdida', color='#e0e0f0', fontsize=12)
        self.ax.tick_params(colors='#8080a0')

        # Ball y trail (inicializar vacíos)
        self.ball, = self.ax.plot([], [], [], 'o', color='#ff4444',
                                  markersize=10, zorder=10)
        self.trail, = self.ax.plot([], [], [], '-', color='#ff6b6b',
                                   linewidth=2, alpha=0.8)

    def update(self, weights, loss):
        """Actualiza la posición de la bola."""
        self.trail_w1.append(weights[0])
        self.trail_w2.append(weights[1])
        self.trail_loss.append(loss)

        self.ball.set_data_3d([weights[0]], [weights[1]], [loss])
        self.trail.set_data_3d(self.trail_w1, self.trail_w2, self.trail_loss)

    def reset(self):
        self.trail_w1.clear()
        self.trail_w2.clear()
        self.trail_loss.clear()


# ═══════════════════════════════════════════════════════════════════════
# Grafo de la Red Neuronal
# ═══════════════════════════════════════════════════════════════════════

class NeuralNetworkGraph:
    """
    Visualización del grafo de la red neuronal.

    Muestra nodos como círculos y conexiones con grosor proporcional
    al valor absoluto del peso, coloreadas por signo (azul=positivo,
    rojo=negativo).
    """

    # Posiciones fijas de los nodos
    LAYOUT = {
        'input': [(0.1, 0.75), (0.1, 0.25)],           # x, y
        'hidden': [(0.45, 0.75), (0.45, 0.25)],         # tanh nodes
        'output': [(0.85, 0.75), (0.85, 0.25)],         # dx/dt, dy/dt
    }
    INPUT_LABELS = ['x', 'y']
    HIDDEN_LABELS = ['tanh', 'tanh']
    OUTPUT_LABELS = ['dx/dt', 'dy/dt']

    # Conexiones: (from_layer, from_idx, to_layer, to_idx, weight_idx)
    CONNECTIONS = [
        ('input', 0, 'hidden', 0, 0),   # x → tanh1: w1
        ('input', 1, 'hidden', 0, 1),   # y → tanh1: w2
        ('input', 0, 'hidden', 1, 2),   # x → tanh2: w3
        ('input', 1, 'hidden', 1, 3),   # y → tanh2: w4
    ]
    BIAS_CONNECTIONS = [
        ('hidden', 0, 4),   # b1
        ('hidden', 1, 5),   # b2
    ]
    WEIGHT_NAMES = ['w1', 'w2', 'w3', 'w4', 'b1', 'b2']

    def __init__(self, ax):
        self.ax = ax
        self.ax.set_xlim(-0.05, 1.05)
        self.ax.set_ylim(-0.05, 1.05)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.set_facecolor('#16162a')
        self.ax.set_title('Red Neuronal', color='#e0e0f0', fontsize=12)

        self.node_patches = {}
        self.conn_lines = []
        self.conn_labels = []
        self.node_labels_obj = []

        self._draw_nodes()
        self._draw_static_connections()

    def _draw_nodes(self):
        node_r = 0.05
        layer_configs = [
            ('input', self.LAYOUT['input'], self.INPUT_LABELS, '#4fc3f7'),
            ('hidden', self.LAYOUT['hidden'], self.HIDDEN_LABELS, '#ffab40'),
            ('output', self.LAYOUT['output'], self.OUTPUT_LABELS, '#00ff88'),
        ]
        for layer_name, positions, labels, color in layer_configs:
            for i, (px, py) in enumerate(positions):
                circle = patches.Circle(
                    (px, py), node_r, fill=True,
                    facecolor=color, edgecolor='white',
                    linewidth=1.5, alpha=0.9, zorder=5
                )
                self.ax.add_patch(circle)
                self.node_patches[(layer_name, i)] = circle
                self.node_labels_obj.append(
                    self.ax.text(px, py, labels[i], ha='center', va='center',
                                fontsize=8, fontweight='bold', color='#0f0f1a',
                                zorder=6)
                )

    def _draw_static_connections(self):
        # Hidden → Output (siempre peso 1, sin entrenar)
        for i in range(2):
            h_pos = self.LAYOUT['hidden'][i]
            o_pos = self.LAYOUT['output'][i]
            line, = self.ax.plot(
                [h_pos[0] + 0.05, o_pos[0] - 0.05],
                [h_pos[1], o_pos[1]],
                '-', color='#4a4a6a', linewidth=1.5, alpha=0.5
            )

    def update(self, weights):
        """Actualiza el grosor y color de las conexiones."""
        # Limpiar conexiones anteriores
        for line in self.conn_lines:
            line.remove()
        for lbl in self.conn_labels:
            lbl.remove()
        self.conn_lines.clear()
        self.conn_labels.clear()

        cmap = plt.cm.RdBu
        norm = colors.Normalize(vmin=-2, vmax=2)

        # Conexiones principales (w1-w4)
        for from_layer, from_idx, to_layer, to_idx, w_idx in self.CONNECTIONS:
            f_pos = self.LAYOUT[from_layer][from_idx]
            t_pos = self.LAYOUT[to_layer][to_idx]
            w = weights[w_idx]
            color = cmap(norm(w))
            lw = max(0.5, min(6.0, abs(w) * 3))

            line, = self.ax.plot(
                [f_pos[0] + 0.05, t_pos[0] - 0.05],
                [f_pos[1], t_pos[1]],
                '-', color=color, linewidth=lw, alpha=0.8, zorder=3
            )
            self.conn_lines.append(line)

            mid_x = (f_pos[0] + t_pos[0]) / 2
            mid_y = (f_pos[1] + t_pos[1]) / 2
            lbl = self.ax.text(
                mid_x, mid_y + 0.04,
                f'{self.WEIGHT_NAMES[w_idx]}={w:+.2f}',
                ha='center', va='bottom', fontsize=7,
                color='#c0c0d0', zorder=7
            )
            self.conn_labels.append(lbl)

        # Bias labels
        for h_idx, b_idx in [(0, 4), (1, 5)]:
            pos = self.LAYOUT['hidden'][h_idx]
            b = weights[b_idx]
            lbl = self.ax.text(
                pos[0], pos[1] - 0.08,
                f'{self.WEIGHT_NAMES[b_idx]}={b:+.2f}',
                ha='center', va='top', fontsize=7,
                color='#ffab40', zorder=7
            )
            self.conn_labels.append(lbl)

    def pulse_forward(self):
        """Efecto de pulso forward (animación simple)."""
        for key, circle in self.node_patches.items():
            circle.set_alpha(1.0)

    def pulse_backward(self):
        """Efecto de pulso backward (animación simple)."""
        for key, circle in self.node_patches.items():
            circle.set_alpha(0.7)


# ═══════════════════════════════════════════════════════════════════════
# Panel de Convergencia en Tiempo Real
# ═══════════════════════════════════════════════════════════════════════

class ConvergencePanel:
    """
    4 subplots que se actualizan en tiempo real durante el entrenamiento:
        1. Pérdida vs época
        2. ||∇L|| vs época
        3. Evolución de pesos
        4. ||ΔW|| vs época
    """

    WEIGHT_COLORS = ['#4fc3f7', '#00ff88', '#ff6b6b', '#e040fb', '#ffab40', '#7c4dff']
    WEIGHT_NAMES = ['w1', 'w2', 'w3', 'w4', 'b1', 'b2']

    def __init__(self, axes):
        """axes: array de 4 matplotlib axes (2×2)."""
        self.axes = axes
        self._style_axes()

        # Data storage
        self.epochs = []
        self.losses = []
        self.grad_norms = []
        self.weight_history = []
        self.weight_deltas = []

        # Line objects for efficient update
        self.loss_line, = axes[0].plot([], [], color='#ff6b6b', lw=2)
        self.grad_line, = axes[1].plot([], [], color='#ffab40', lw=2)
        self.weight_lines = []
        for i, (name, color) in enumerate(zip(self.WEIGHT_NAMES, self.WEIGHT_COLORS)):
            line, = axes[2].plot([], [], color=color, lw=1.5, label=name)
            self.weight_lines.append(line)
        axes[2].legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#3a3a5a',
                       labelcolor='#c0c0d0', ncol=3, loc='upper right')
        self.delta_line, = axes[3].plot([], [], color='#7c4dff', lw=2)

    def _style_axes(self):
        titles = ['Pérdida (Loss)', '||∇L||', 'Pesos', '||ΔW||']
        for ax, title in zip(self.axes, titles):
            ax.set_facecolor('#16162a')
            ax.set_title(title, color='#e0e0f0', fontsize=10)
            ax.tick_params(colors='#8080a0', labelsize=8)
            ax.grid(True, color='#1e1e3a', alpha=0.5)
            for spine in ax.spines.values():
                spine.set_color('#2a2a4a')

    def update(self, epoch, loss, gradients, weights, weight_delta):
        """Agregar un punto de datos y re-dibujar."""
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.grad_norms.append(np.linalg.norm(gradients))
        self.weight_history.append(weights.copy())
        self.weight_deltas.append(weight_delta)

        ep = self.epochs

        # 1. Loss
        self.loss_line.set_data(ep, self.losses)
        self.axes[0].relim()
        self.axes[0].autoscale_view()

        # 2. Gradient norm
        self.grad_line.set_data(ep, self.grad_norms)
        self.axes[1].relim()
        self.axes[1].autoscale_view()

        # 3. Weights
        wa = np.array(self.weight_history)
        for i, line in enumerate(self.weight_lines):
            line.set_data(ep, wa[:, i])
        self.axes[2].relim()
        self.axes[2].autoscale_view()

        # 4. Weight deltas
        self.delta_line.set_data(ep, self.weight_deltas)
        self.axes[3].relim()
        self.axes[3].autoscale_view()

    def reset(self):
        self.epochs.clear()
        self.losses.clear()
        self.grad_norms.clear()
        self.weight_history.clear()
        self.weight_deltas.clear()


# ═══════════════════════════════════════════════════════════════════════
# Trayectoria 2D Dinámica + Campo Vectorial
# ═══════════════════════════════════════════════════════════════════════

class DynamicTrajectoryPlot:
    """
    Trayectoria 2D animada con campo vectorial y comparación target.
    """

    def __init__(self, ax, neural_field, plot_range=(-3, 3), grid_size=15):
        self.ax = ax
        self.field = neural_field
        self.plot_range = plot_range
        self.grid_size = grid_size

        ax.set_facecolor('#16162a')
        ax.set_xlim(plot_range)
        ax.set_ylim(plot_range)
        ax.set_aspect('equal')
        ax.set_xlabel('x', color='#c0c0d0')
        ax.set_ylabel('y', color='#c0c0d0')
        ax.set_title('Trayectoria 2D', color='#e0e0f0', fontsize=12)
        ax.tick_params(colors='#8080a0')
        ax.grid(True, color='#1e1e3a', alpha=0.5)
        for spine in ax.spines.values():
            spine.set_color('#2a2a4a')

        self.target_line, = ax.plot([], [], '--', color='#ff6b6b',
                                    lw=2, alpha=0.6, label='Objetivo')
        self.pred_line, = ax.plot([], [], '-', color='#00d4ff',
                                  lw=2, label='Predicha')
        self.start_marker, = ax.plot([], [], '*', color='#00ff88',
                                     ms=15, zorder=5)
        self.streamplot_obj = None

        ax.legend(fontsize=9, facecolor='#1a1a2e', edgecolor='#3a3a5a',
                  labelcolor='#c0c0d0', loc='upper right')

    def set_target(self, xs, ys):
        self.target_line.set_data(xs, ys)

    def update(self, xs_pred, ys_pred):
        self.pred_line.set_data(xs_pred, ys_pred)
        if len(xs_pred) > 0:
            self.start_marker.set_data([xs_pred[0]], [ys_pred[0]])

    def update_field(self):
        """Redibuja el campo vectorial (costoso — llamar poco)."""
        if self.streamplot_obj:
            self.streamplot_obj.lines.remove()
            for art in self.ax.get_children():
                if isinstance(art, patches.FancyArrowPatch):
                    art.remove()

        xr = np.linspace(*self.plot_range, self.grid_size)
        yr = np.linspace(*self.plot_range, self.grid_size)
        X, Y = np.meshgrid(xr, yr)
        U, V = self.field.get_field(X, Y)
        speed = np.sqrt(U**2 + V**2)

        try:
            self.streamplot_obj = self.ax.streamplot(
                X, Y, U, V, color=speed, cmap='viridis',
                density=1.0, linewidth=0.6, arrowsize=0.8
            )
        except Exception:
            pass

    def reset(self):
        self.pred_line.set_data([], [])


# ═══════════════════════════════════════════════════════════════════════
# Heatmap de Activaciones Dinámico
# ═══════════════════════════════════════════════════════════════════════

class DynamicActivationHeatmap:
    """Heatmaps de activaciones f_θ y g_θ que se actualizan."""

    def __init__(self, axes, neural_field, plot_range=(-3, 3), resolution=50):
        """axes: lista de 2 axes (componente x, componente y)."""
        self.axes = axes
        self.field = neural_field
        self.plot_range = plot_range
        self.resolution = resolution

        ext = [plot_range[0], plot_range[1], plot_range[0], plot_range[1]]
        self.images = []
        titles = ['f_θ → dx/dt', 'g_θ → dy/dt']

        for ax, title in zip(axes, titles):
            ax.set_facecolor('#16162a')
            dummy = np.zeros((resolution, resolution))
            im = ax.imshow(dummy, extent=ext, origin='lower',
                           cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
            ax.set_title(title, color='#e0e0f0', fontsize=10)
            ax.tick_params(colors='#8080a0', labelsize=7)
            self.images.append(im)

    def update(self):
        xr = np.linspace(*self.plot_range, self.resolution)
        yr = np.linspace(*self.plot_range, self.resolution)
        X, Y = np.meshgrid(xr, yr)
        U, V = self.field.get_field(X, Y)
        self.images[0].set_data(U)
        self.images[1].set_data(V)
