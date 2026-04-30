import matplotlib.pyplot as plt
import config

from core.neural_field import CampoNeuronal
from core.ode_solver import ODE_solucionador

# se crea un campo
field = CampoNeuronal()

# se crea un solucionador
solucionador = ODE_solucionador()

# se resuelve el sistema dinamico
solucion = solucionador.resolver(field, config.ESTADO_INICIAL)

# se extraen todos los valores de x
x = solucion.y[0]

# se extraen todos los valores de y
y = solucion.y[1]

# esta parte va en static_plot.py
# abre una pestaña que grafica
plt.figure(figsize=(6,6))

# grafica la trayectoria de los puntos
plt.plot(x, y)

# marca el inicio 
plt.scatter(x[0], y[0], label="Inicio")

# marca el final
plt.scatter(x[-1], y[-1], label="Fin")

# etiquetas
plt.xlabel("x")
plt.ylabel("y")

# titulo
plt.title("Trayectoria del sistema dinámico")

# hace que 1 unidad en x sea igual a 1 unidad en y
plt.axis("equal")

# muestra el inicio y el final 
plt.legend()

# muestra la cudricula
plt.grid()

# muestra la ventana de la grafica
plt.show()