"""
config.py — Parámetros globales del Neural ODE Simulator.

Centraliza todas las constantes del sistema: tiempo de simulación,
condiciones iniciales, pesos por defecto, hiperparámetros de
entrenamiento y configuración de ventana.
"""

# ═══════════════════════════════════════════════════════════════════════
# SIMULACIÓN — Intervalo de tiempo y resolución
# ═══════════════════════════════════════════════════════════════════════
T_START = 0.0       # Tiempo inicial (s)
T_END = 10.0        # Tiempo final (s)
DT = 0.05           # Paso de tiempo Δt (s)
N_STEPS = int((T_END - T_START) / DT)  # Número total de pasos

# ═══════════════════════════════════════════════════════════════════════
# CONDICIONES INICIALES — Posición de arranque del punto
# ═══════════════════════════════════════════════════════════════════════
X0 = 1.0            # Posición inicial en x
Y0 = 0.0            # Posición inicial en y

# ═══════════════════════════════════════════════════════════════════════
# RED NEURONAL — Pesos por defecto (generan un círculo)
# ═══════════════════════════════════════════════════════════════════════
# θ = [w1, w2, w3, w4, b1, b2]
# f_θ(x,y) = tanh(w1·x + w2·y + b1)   → dx/dt
# g_θ(x,y) = tanh(w3·x + w4·y + b2)   → dy/dt
DEFAULT_WEIGHTS = [0.0, -1.0, 1.0, 0.0, 0.0, 0.0]

WEIGHT_NAMES = ['w1', 'w2', 'w3', 'w4', 'b1', 'b2']

# ═══════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO — Hiperparámetros del descenso de gradiente
# ═══════════════════════════════════════════════════════════════════════
LEARNING_RATE = 0.01        # Tasa de aprendizaje η
MAX_EPOCHS = 500            # Máximo de épocas de entrenamiento
GRADIENT_EPSILON = 1e-4     # ε para diferencias finitas
BATCH_SIZE = 32             # Tamaño del lote (para uso futuro)

# ═══════════════════════════════════════════════════════════════════════
# VISUALIZACIÓN — Configuración de gráficas y ventana
# ═══════════════════════════════════════════════════════════════════════
WINDOW_WIDTH = 1400         # Ancho de la ventana principal (px)
WINDOW_HEIGHT = 900         # Alto de la ventana principal (px)
FPS_TARGET = 30             # FPS objetivo para animaciones

# Rango del plano 2D para visualización
PLOT_RANGE = (-3.0, 3.0)   # Límites de x e y en las gráficas
FIELD_GRID_SIZE = 20        # Resolución de la malla del campo vectorial

# Colores del sistema (paleta oscura moderna)
COLORS = {
    'background': '#0f0f1a',
    'grid': '#1a1a2e',
    'trajectory': '#00d4ff',
    'target': '#ff6b6b',
    'start_point': '#00ff88',
    'end_point': '#ff4444',
    'field_arrows': '#4a4a6a',
    'loss_surface': 'viridis',
    'activation_cmap': 'RdBu',
    'positive_weight': '#4fc3f7',
    'negative_weight': '#ef5350',
}
