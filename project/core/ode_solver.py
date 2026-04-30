import numpy as np

# ayuda a resolver las ecuaciones diferenciales numericamente
from scipy.integrate import solve_ivp
import config

# esta clase resuelve el sistema dinamico, calcula trayectorias y simula el movimiento del punto
class ODE_solucionador:

    # esta funcion hace la simulacion con diferentes parametros
    # el campo vectorial, el estado inicial que es punto de inicio, y el tiempo maximo de simulacion
    def resolver(self, campo, estado_inicial,  t_max = None, n_puntos = None):
        

        # estos valores pueden venir por defecto o
        # o se pueden remplazar si se necesita
        t_max = t_max or config.T_MAX
        n_puntos = n_puntos or config.N_PUNTOS

        # crea valores 1000 valores entre 0 y t_max
        tiempos = np.linspace(0, t_max, 1000)

        # calcula la trayectoria recibiendo la funcion velocidad que define el movimiento, 
        # el intervalo 0 y t_max, el punto inicial del sistema y los tiempos correpondientes
        solucion = solve_ivp(campo.velocidad, [0, t_max], estado_inicial, t_eval = tiempos)

        return solucion
