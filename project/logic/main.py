from LinealEquation import LinealEquation
from SquareEquation import SquareEquation
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
     #y = 2 * x +5 # lo que queremos que aprenda
     # -2,-1,0,1,2 Quedan Puntos en el Plano: (-1,-2) (-0.5,-1)
                 #(0,0) (0,5,1) (1,2)
     #Metodo random.randn() genera valores 0 <= numero < 1 tipo 0.1,0.3
     w = np.random.randn()  # peso
     b = np.random.randn()  # bias de la red(direcciones)
     z = np.random.randn()
    #No se ponen valores como 0,1,2 porque la red neuronal no aprenderia correctamente   
     # Datos para red que aprende usando ypred = wx + b
     y=2*x**2+3*x+6
     linealF = SquareEquation(x,y,w,b,z)
     linealF.prueba()
     
    