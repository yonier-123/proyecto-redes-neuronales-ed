import numpy as np

# ejemplo de un sistemas dinamico que controla el movimiento de un punto en el plano
class CampoNeuronal:

    # recibe valores por parametro, para luego poder crear presets
    def __init__(self, w1=1.0, w2=-1.0, w3=1.0, w4=1.0, b1=0.0, b2=0.0):

        # pesos
        # estos valores influyen en como se maneja el comportamiento del sistema en x e y
        # se representan matematicamente como una matriz 2x2

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        
        # biases
        # son desplazamientos adicionales en las ecuaciones

        self.b1 = b1
        self.b2 = b2

# esta funcion recibe el tiempo y posicion actual del campo vectorial
# y devuelve la velocidad de cambio
    def velocidad(self, t, estado_actual):
        x, y = estado_actual
        # formulas:

        # ecuacion diferencial para x
        # dx/dt = f0(x,y), f0(x,y) = tanh((w1 * x) + (w2 * y) + b1) 
        dx = np.tanh(
            self.w1*x + 
            self.w2*y + 
            self.b1)
        
        # ecuacion diferencial para y
        # dy/dt = g0(x,y), g0(x,y) = tanh((w3 * x) + (w4 * y) + b2) 
        dy = np.tanh(
            self.w3*x + 
            self.w4*y + 
            self.b2)

        # devuelve los valores de dx y dy
        return [dx, dy]