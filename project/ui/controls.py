"""
controls.py — Interfaz gráfica interactiva (MVP) con PyQt5.

Layout:
    ┌──────────┬───────────────┬──────────────┐
    │ Red      │ Trayectoria   │ Paisaje 3D   │
    │ Neuronal │ 2D + Campo    │ de Pérdida   │
    │          │               │              │
    ├──────────┤ Heatmaps de   ├──────────────┤
    │ Controles│ Activaciones  │ Convergencia │
    │ y Params │               │ (4 plots)    │
    └──────────┴───────────────┴──────────────┘

Controles:
    - PLAY / PAUSE / RESET
    - Slider Learning Rate (log scale)
    - ComboBox Optimizador, Preset, Solver
    - Display: Epoch, Loss
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QComboBox, QSlider, QGroupBox, QFrame,
    QApplication, QStyleFactory,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from core.neural_field import NeuralField
from core.ode_solver import ODESolver
from core.trainer import GradientTrainer
from simulation.presets import get_preset, list_presets
from simulation.optimizers import create_optimizer
from visualization.animator import (
    LossLandscape3D,
    NeuralNetworkGraph,
    ConvergencePanel,
    DynamicTrajectoryPlot,
    DynamicActivationHeatmap,
)
import config


# ═══════════════════════════════════════════════════════════════════════
# Hilo de entrenamiento (no bloquea la UI)
# ═══════════════════════════════════════════════════════════════════════

class TrainingThread(QThread):
    """Hilo que ejecuta el entrenamiento paso a paso."""
    epoch_done = pyqtSignal(int, float, object, object, float)  # epoch, loss, grads, weights, delta
    training_finished = pyqtSignal()

    def __init__(self, trainer, x0, y0, t_span, dt, xs_target, ys_target,
                 delay_ms=50):
        super().__init__()
        self.trainer = trainer
        self.x0, self.y0 = x0, y0
        self.t_span, self.dt = t_span, dt
        self.xs_target = xs_target
        self.ys_target = ys_target
        self.delay_ms = delay_ms
        self.running = False
        self._epoch = 0

    def run(self):
        self.running = True
        while self.running and self._epoch < config.MAX_EPOCHS:
            loss, grads = self.trainer.step(
                self.x0, self.y0, self.t_span, self.dt,
                self.xs_target, self.ys_target
            )
            weights = self.trainer.field.get_weights()
            delta = self.trainer.history['weight_deltas'][-1]
            self._epoch += 1
            self.epoch_done.emit(self._epoch, loss, grads, weights, delta)

            if loss < 1e-6:
                break

            self.msleep(self.delay_ms)

        self.training_finished.emit()

    def stop(self):
        self.running = False

    @property
    def epoch(self):
        return self._epoch


# ═══════════════════════════════════════════════════════════════════════
# Canvas de Matplotlib embebido en PyQt5
# ═══════════════════════════════════════════════════════════════════════

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, nrows=1, ncols=1, figsize=(6, 4), projection=None):
        self.fig = Figure(figsize=figsize, facecolor='#0f0f1a')
        if projection == '3d':
            self.axes = [self.fig.add_subplot(nrows, ncols, 1,
                                              projection='3d')]
        else:
            self.axes = []
            for i in range(nrows * ncols):
                ax = self.fig.add_subplot(nrows, ncols, i + 1)
                self.axes.append(ax)
        self.fig.tight_layout(pad=1.5)
        super().__init__(self.fig)


# ═══════════════════════════════════════════════════════════════════════
# Ventana principal del MVP
# ═══════════════════════════════════════════════════════════════════════

class NeuralODEApp(QMainWindow):
    """Ventana principal de la aplicación Neural ODE Simulator."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('🧠 Neural ODE Simulator — Ecuaciones Diferenciales')
        self.resize(config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        self._apply_dark_theme()

        # Backend state
        self.field = NeuralField()
        self.solver = ODESolver(self.field)
        self.optimizer = create_optimizer('sgd', lr=config.LEARNING_RATE)
        self.trainer = GradientTrainer(self.field, self.solver, self.optimizer)
        self.training_thread = None
        self.current_preset = 'circulo'

        # Target trajectory
        self.xs_target = None
        self.ys_target = None
        self.t_span = (config.T_START, min(config.T_END, 5.0))

        self._build_ui()
        self._load_preset('circulo')

    # ─── Dark Theme ───────────────────────────────────────────────

    def _apply_dark_theme(self):
        palette = QPalette()
        bg = QColor('#0f0f1a')
        text = QColor('#e0e0f0')
        accent = QColor('#2a2a4a')
        palette.setColor(QPalette.Window, bg)
        palette.setColor(QPalette.WindowText, text)
        palette.setColor(QPalette.Base, QColor('#16162a'))
        palette.setColor(QPalette.AlternateBase, accent)
        palette.setColor(QPalette.Text, text)
        palette.setColor(QPalette.Button, accent)
        palette.setColor(QPalette.ButtonText, text)
        palette.setColor(QPalette.Highlight, QColor('#4fc3f7'))
        palette.setColor(QPalette.HighlightedText, QColor('#000000'))
        self.setPalette(palette)
        self.setStyleSheet("""
            QMainWindow { background-color: #0f0f1a; }
            QGroupBox {
                border: 1px solid #2a2a4a;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 14px;
                color: #e0e0f0;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #2a2a4a;
                border: 1px solid #3a3a5a;
                border-radius: 6px;
                padding: 8px 16px;
                color: #e0e0f0;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #3a3a5a;
                border-color: #4fc3f7;
            }
            QPushButton:pressed { background-color: #4fc3f7; color: #000; }
            QPushButton#playBtn { border-color: #00ff88; color: #00ff88; }
            QPushButton#playBtn:hover { background-color: #00ff88; color: #000; }
            QPushButton#pauseBtn { border-color: #ffab40; color: #ffab40; }
            QPushButton#pauseBtn:hover { background-color: #ffab40; color: #000; }
            QPushButton#resetBtn { border-color: #ff6b6b; color: #ff6b6b; }
            QPushButton#resetBtn:hover { background-color: #ff6b6b; color: #000; }
            QComboBox {
                background-color: #1a1a2e;
                border: 1px solid #3a3a5a;
                border-radius: 4px;
                padding: 4px 8px;
                color: #e0e0f0;
            }
            QComboBox::drop-down { border: none; }
            QSlider::groove:horizontal {
                height: 6px;
                background: #2a2a4a;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4fc3f7;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QLabel { color: #c0c0d0; }
        """)

    # ─── Build UI ─────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ── Panel izquierdo: Red + Controles ──
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)

        # Red neuronal
        self.nn_canvas = MplCanvas(1, 1, figsize=(4, 3))
        self.nn_graph = NeuralNetworkGraph(self.nn_canvas.axes[0])
        left_layout.addWidget(self.nn_canvas, stretch=3)

        # Controles
        left_layout.addWidget(self._build_controls(), stretch=4)
        splitter.addWidget(left_widget)

        # ── Panel central: Trayectoria + Heatmaps ──
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(4, 4, 4, 4)

        self.traj_canvas = MplCanvas(1, 1, figsize=(6, 5))
        self.traj_plot = DynamicTrajectoryPlot(
            self.traj_canvas.axes[0], self.field
        )
        center_layout.addWidget(self.traj_canvas, stretch=3)

        self.heat_canvas = MplCanvas(1, 2, figsize=(6, 2.5))
        self.heat_map = DynamicActivationHeatmap(
            self.heat_canvas.axes, self.field
        )
        center_layout.addWidget(self.heat_canvas, stretch=2)
        splitter.addWidget(center_widget)

        # ── Panel derecho: Loss 3D + Convergencia ──
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)

        self.loss3d_canvas = MplCanvas(1, 1, figsize=(5, 4), projection='3d')
        right_layout.addWidget(self.loss3d_canvas, stretch=3)

        self.conv_canvas = MplCanvas(2, 2, figsize=(5, 4))
        self.conv_panel = ConvergencePanel(self.conv_canvas.axes)
        right_layout.addWidget(self.conv_canvas, stretch=3)
        splitter.addWidget(right_widget)

        splitter.setSizes([300, 500, 400])

        # Status bar
        self.status_label = QLabel('  Listo — Selecciona un preset y presiona PLAY')
        self.status_label.setStyleSheet(
            'color: #00ff88; font-size: 12px; padding: 4px;'
        )
        self.statusBar().addWidget(self.status_label)
        self.statusBar().setStyleSheet('background-color: #0a0a14; border-top: 1px solid #2a2a4a;')

    def _build_controls(self):
        group = QGroupBox('Controles')
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        # ── Botones PLAY / PAUSE / RESET ──
        btn_layout = QHBoxLayout()
        self.play_btn = QPushButton('▶ PLAY')
        self.play_btn.setObjectName('playBtn')
        self.play_btn.clicked.connect(self._on_play)
        btn_layout.addWidget(self.play_btn)

        self.pause_btn = QPushButton('⏸ PAUSE')
        self.pause_btn.setObjectName('pauseBtn')
        self.pause_btn.clicked.connect(self._on_pause)
        self.pause_btn.setEnabled(False)
        btn_layout.addWidget(self.pause_btn)

        self.reset_btn = QPushButton('⟲ RESET')
        self.reset_btn.setObjectName('resetBtn')
        self.reset_btn.clicked.connect(self._on_reset)
        btn_layout.addWidget(self.reset_btn)
        layout.addLayout(btn_layout)

        # ── Preset ──
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel('Preset:'))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list_presets())
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self.preset_combo)
        layout.addLayout(preset_layout)

        # ── Optimizador ──
        opt_layout = QHBoxLayout()
        opt_layout.addWidget(QLabel('Optimizador:'))
        self.opt_combo = QComboBox()
        self.opt_combo.addItems(['sgd', 'momentum', 'adam'])
        self.opt_combo.currentTextChanged.connect(self._on_optimizer_changed)
        opt_layout.addWidget(self.opt_combo)
        layout.addLayout(opt_layout)

        # ── Solver ──
        solver_layout = QHBoxLayout()
        solver_layout.addWidget(QLabel('Solver:'))
        self.solver_combo = QComboBox()
        self.solver_combo.addItems(['euler', 'rk4'])
        solver_layout.addWidget(self.solver_combo)
        layout.addLayout(solver_layout)

        # ── Learning Rate ──
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel('Learning Rate:'))
        self.lr_label = QLabel(f'{config.LEARNING_RATE:.4f}')
        self.lr_label.setStyleSheet('color: #4fc3f7; font-weight: bold;')
        lr_layout.addWidget(self.lr_label)
        layout.addLayout(lr_layout)

        self.lr_slider = QSlider(Qt.Horizontal)
        self.lr_slider.setRange(0, 100)
        self.lr_slider.setValue(50)
        self.lr_slider.valueChanged.connect(self._on_lr_changed)
        layout.addWidget(self.lr_slider)

        # ── Display métricas ──
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet('color: #2a2a4a;')
        layout.addWidget(sep)

        self.epoch_label = QLabel('Época: 0')
        self.epoch_label.setStyleSheet('color: #4fc3f7; font-size: 14px; font-weight: bold;')
        layout.addWidget(self.epoch_label)

        self.loss_label = QLabel('Loss: —')
        self.loss_label.setStyleSheet('color: #ff6b6b; font-size: 14px; font-weight: bold;')
        layout.addWidget(self.loss_label)

        self.weights_label = QLabel('Pesos: —')
        self.weights_label.setStyleSheet('color: #c0c0d0; font-size: 10px;')
        self.weights_label.setWordWrap(True)
        layout.addWidget(self.weights_label)

        layout.addStretch()
        return group

    # ─── Preset & Config ─────────────────────────────────────────

    def _load_preset(self, name):
        self.current_preset = name
        preset = get_preset(name)

        # Generar trayectoria objetivo
        target_field = NeuralField(preset['weights'])
        target_solver = ODESolver(target_field)
        _, self.xs_target, self.ys_target = target_solver.solve_euler(
            preset['x0'], preset['y0'], self.t_span, config.DT
        )

        # Mostrar en trayectoria 2D
        self.traj_plot.set_target(self.xs_target, self.ys_target)

        # Randomizar pesos de la red actual
        self.field.randomize_weights(scale=0.5, rng=np.random.default_rng())
        self.nn_graph.update(self.field.get_weights())
        self.heat_map.update()

        # Actualizar displays
        w = self.field.get_weights()
        w_str = '  '.join(f'{n}={v:+.3f}' for n, v in
                          zip(NeuralField.WEIGHT_NAMES, w))
        self.weights_label.setText(f'Pesos: {w_str}')
        self.status_label.setText(f'  Preset: {name} — {preset["descripcion"]}')

        # Rebuild loss landscape
        self._build_loss_landscape(preset)

        # Redraw
        self.nn_canvas.draw_idle()
        self.traj_canvas.draw_idle()
        self.heat_canvas.draw_idle()

    def _build_loss_landscape(self, preset):
        """Recalcula y dibuja la superficie 3D de pérdida."""
        self.loss3d_canvas.fig.clear()
        ax3d = self.loss3d_canvas.fig.add_subplot(111, projection='3d')
        self.loss3d_canvas.axes = [ax3d]

        # Guardar pesos actuales
        w_current = self.field.get_weights()

        self.loss_landscape = LossLandscape3D(
            ax3d, self.field, self.solver,
            get_preset(self.current_preset)['x0'],
            get_preset(self.current_preset)['y0'],
            self.t_span, config.DT,
            self.xs_target, self.ys_target,
            w_range=(-2, 2), resolution=25
        )

        # Restaurar pesos
        self.field.set_weights(w_current)
        self.loss3d_canvas.draw_idle()

    # ─── Callbacks ────────────────────────────────────────────────

    def _on_play(self):
        if self.training_thread and self.training_thread.isRunning():
            return

        preset = get_preset(self.current_preset)
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.preset_combo.setEnabled(False)

        # Rebuild optimizer with current LR
        lr = self._get_lr()
        opt_name = self.opt_combo.currentText()
        self.optimizer = create_optimizer(opt_name, lr=lr)
        self.trainer = GradientTrainer(
            self.field, self.solver, self.optimizer
        )

        self.training_thread = TrainingThread(
            self.trainer,
            preset['x0'], preset['y0'],
            self.t_span, config.DT,
            self.xs_target, self.ys_target,
            delay_ms=30
        )
        self.training_thread.epoch_done.connect(self._on_epoch_done)
        self.training_thread.training_finished.connect(self._on_training_finished)
        self.training_thread.start()

        self.status_label.setText('  ⏵ Entrenando...')

    def _on_pause(self):
        if self.training_thread:
            self.training_thread.stop()
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.status_label.setText('  ⏸ Pausado — Ajusta parámetros y presiona PLAY')

    def _on_reset(self):
        if self.training_thread:
            self.training_thread.stop()
            self.training_thread.wait()

        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.preset_combo.setEnabled(True)

        # Reset convergence
        self.conv_panel.reset()
        self.loss_landscape.reset()
        self.traj_plot.reset()

        # Reset UI
        self.epoch_label.setText('Época: 0')
        self.loss_label.setText('Loss: —')

        # Reload preset
        self._load_preset(self.current_preset)

        # Redraw all
        for canvas in [self.nn_canvas, self.traj_canvas, self.heat_canvas,
                       self.loss3d_canvas, self.conv_canvas]:
            canvas.draw_idle()

        self.status_label.setText('  ⟲ Reiniciado — Listo para entrenar')

    def _on_preset_changed(self, name):
        if self.training_thread and self.training_thread.isRunning():
            return
        self.conv_panel.reset()
        self._load_preset(name)
        for canvas in [self.nn_canvas, self.traj_canvas, self.heat_canvas,
                       self.loss3d_canvas, self.conv_canvas]:
            canvas.draw_idle()

    def _on_optimizer_changed(self, name):
        lr = self._get_lr()
        self.optimizer = create_optimizer(name, lr=lr)
        self.status_label.setText(f'  Optimizador: {name.upper()} (lr={lr:.4f})')

    def _on_lr_changed(self, value):
        lr = self._get_lr()
        self.lr_label.setText(f'{lr:.4f}')
        self.optimizer.lr = lr

    def _get_lr(self):
        """Convierte slider (0-100) a LR en escala logarítmica."""
        v = self.lr_slider.value()
        return 10 ** (-4 + v * 3 / 100)  # 0.0001 → 0.1

    # ─── Epoch update (from training thread) ──────────────────────

    def _on_epoch_done(self, epoch, loss, grads, weights, delta):
        """Actualiza todas las visualizaciones con datos de la época."""
        # Métricas
        self.epoch_label.setText(f'Época: {epoch}')
        self.loss_label.setText(f'Loss: {loss:.6f}')
        w_str = '  '.join(f'{n}={v:+.3f}' for n, v in
                          zip(NeuralField.WEIGHT_NAMES, weights))
        self.weights_label.setText(f'Pesos: {w_str}')

        # 1. Red neuronal
        self.nn_graph.update(weights)
        self.nn_canvas.draw_idle()

        # 2. Convergencia
        self.conv_panel.update(epoch, loss, grads, weights, delta)
        self.conv_canvas.draw_idle()

        # 3. Paisaje 3D
        self.loss_landscape.update(weights, loss)
        self.loss3d_canvas.draw_idle()

        # 4. Trayectoria predicha (cada 5 épocas)
        if epoch % 5 == 0:
            preset = get_preset(self.current_preset)
            _, xs_p, ys_p = self.solver.solve_euler(
                preset['x0'], preset['y0'], self.t_span, config.DT
            )
            self.traj_plot.update(xs_p, ys_p)
            self.traj_canvas.draw_idle()

        # 5. Heatmap (cada 10 épocas)
        if epoch % 10 == 0:
            self.heat_map.update()
            self.heat_canvas.draw_idle()

    def _on_training_finished(self):
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.preset_combo.setEnabled(True)
        self.status_label.setText('  ✓ Entrenamiento completado')

    # ─── Cleanup ──────────────────────────────────────────────────

    def closeEvent(self, event):
        if self.training_thread:
            self.training_thread.stop()
            self.training_thread.wait()
        event.accept()
