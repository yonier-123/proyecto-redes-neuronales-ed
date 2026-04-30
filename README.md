# Neural ODE Simulator

Simulador interactivo que visualiza el aprendizaje de redes neuronales
como un sistema dinamico regido por Ecuaciones Diferenciales Ordinarias (EDO).

## Que es esto?

Este proyecto demuestra la conexion profunda entre **redes neuronales** y
**ecuaciones diferenciales**. El descenso de gradiente ES una EDO:

```
dW/dt = -n * grad(L(W))    (EDO en el espacio de pesos)
dz/dt = F_theta(z)          (EDO en el espacio de estados)
```

La aplicacion resuelve ambas EDOs numericamente y visualiza todo en tiempo real.

## Visualizaciones

| # | Visualizacion | Descripcion |
|---|---------------|-------------|
| 1 | Trayectoria 2D + Campo Vectorial | Streamplot con trayectoria degradada por tiempo |
| 2 | Paisaje de Perdida 3D | Superficie w1 vs w2 vs Loss con bola animada |
| 3 | Convergencia (4 plots) | Loss, norma del gradiente, pesos, tasa de cambio |
| 4 | Heatmap de Activaciones | Mapas tanh(w1*x + w2*y + b) para dx/dt y dy/dt |
| 5 | Red Neuronal Animada | Grafo con conexiones coloreadas por peso |

## Requisitos

- Python 3.10+
- numpy, scipy, matplotlib, PyQt5, pytest

## Instalacion

```bash
git clone https://github.com/yonier-123/proyecto-redes-neuronales-ed.git
cd proyecto-redes-neuronales-ed/project
pip install -r requirements.txt
```

## Uso

```bash
# Aplicacion interactiva (MVP) - default
python main.py

# Demos estaticas por consola
python main.py --mode all                           # Panel con todos los presets
python main.py --mode static --preset circulo       # Un preset con campo vectorial
python main.py --mode train --optimizer adam         # Entrenamiento con Adam
python main.py --mode activations                   # Heatmaps de activaciones

# Opciones de entrenamiento
python main.py --mode train --preset espiral_entrada --optimizer momentum --lr 0.05 --epochs 300
```

## Controles de la UI

| Control | Funcion |
|---------|---------|
| PLAY | Inicia el entrenamiento |
| PAUSE | Pausa el entrenamiento |
| RESET | Reinicia pesos y graficas |
| Preset | circulo, espiral_entrada, espiral_salida, punto_fijo, oscilacion, caos |
| Optimizador | SGD, Momentum, Adam |
| Solver | Euler, RK4 |
| Learning Rate | Slider logaritmico (0.0001 - 0.1) |
| Velocidad | 1x - 10x |
| Pesos manuales | Sliders para w1, w2, w3, w4, b1, b2 |
| EXPORT PNG | Guarda las 5 graficas como imagenes |

## Estructura del proyecto

```
project/
  config.py                  Parametros globales
  main.py                    Punto de entrada CLI + GUI
  requirements.txt           Dependencias
  core/
    neural_field.py           Red neuronal como campo de velocidades
    ode_solver.py             Solvers: Euler + RK45 (scipy)
    trainer.py                Descenso de gradiente por diferencias finitas
  simulation/
    presets.py                6 configuraciones predefinidas
    optimizers.py             SGD, Momentum, Adam
  visualization/
    static_plot.py            Graficas estaticas (matplotlib)
    animator.py               Componentes animados para la UI
  ui/
    controls.py               Interfaz PyQt5 con tema oscuro
  tests/
    test_solver.py            33 tests unitarios
    test_integration.py       12 tests de integracion
```

## Modelo matematico

La red neuronal de 6 parametros define un campo de velocidades:

```
dx/dt = tanh(w1*x + w2*y + b1)
dy/dt = tanh(w3*x + w4*y + b2)
```

El entrenamiento minimiza la perdida (MSE entre trayectoria predicha y objetivo)
usando descenso de gradiente por diferencias finitas:

```
grad(L)/grad(wi) ~ [L(w + eps*ei) - L(w)] / eps
```

## Presets disponibles

| Preset | Pesos | Comportamiento |
|--------|-------|----------------|
| circulo | [0, -1, 1, 0, 0, 0] | Rotacion pura |
| espiral_entrada | [-0.1, -1, 1, -0.1, 0, 0] | Espiral convergente |
| espiral_salida | [0.1, -1, 1, 0.1, 0, 0] | Espiral divergente |
| punto_fijo | [-1, 0, 0, -1, 0, 0] | Nodo estable |
| oscilacion | [-0.2, -0.8, 0.8, -0.2, 0.1, -0.1] | Oscilacion amortiguada |
| caos | [0.5, -1.2, 1.3, 0.3, 0.2, -0.1] | Comportamiento complejo |

## Tests

```bash
# Tests unitarios (33 tests)
python -m pytest tests/test_solver.py -v

# Tests de integracion (12 tests)
python -m pytest tests/test_integration.py -v

# Todos los tests
python -m pytest tests/ -v
```

## Conexion con Ecuaciones Diferenciales

El proyecto demuestra 2 EDOs simultaneas:

1. **Sistema dinamico** (la trayectoria): `dz/dt = F_theta(z)` - EDO en el espacio de estados
2. **Aprendizaje** (el entrenamiento): `dW/dt = -grad(L(W))` - EDO en el espacio de pesos

Ambas se resuelven numericamente con los mismos metodos:
- **Euler explicito**: Simple pero puede divergir con LR grande
- **Runge-Kutta 4**: Mas estable y preciso

Referencia: Chen et al. (2018) "Neural Ordinary Differential Equations" - NeurIPS 2018.

## Autores

Proyecto de Ecuaciones Diferenciales - Universidad
