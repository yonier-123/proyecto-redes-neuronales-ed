from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np#Para realizar calculos(Ayuda para Graficar)

#Importaciones para comunicarme con las clases de las redes neuronales programadas
from logic.LinealEquation import LinealEquation#Para comunicarme con la clase LinealEquation
from logic.SquareEquation import SquareEquation#Para comunicarme con la red que me genera una cuadratica


def crearGraphics(ventana):
    global objLineal
    global x,y,w,b,z
    global grafica,canvas

    #Crear Grafica
    figura = Figure(figsize=(5,4),dpi=100)

    grafica = figura.add_subplot(111)
    x = np.array([-1, -0.5, 0, 0.5, 1])#Datos de entrada red neuronal
    #y = y = 2 * x +5
    #grafica.plot(x,y)

   

    grafica.legend()
    canvas = FigureCanvasTkAgg(figura,master=ventana)
    canvas.draw()
    return canvas

def actualizarGrafica(tipoFuncion):

    grafica.clear()#Limpia Grafica para poner otra 
   
    x = np.array([-1,-0.5,0,0.5,1])#Valores para la red neuronal en x

    if tipoFuncion == "FuncionLineal":
        w = np.random.randn()  # peso
        b = np.random.randn()  # bias de la red(direcciones)
        y = 2*x + 5
        grafica.plot(x, y)
        #Comunicaciones de Objetos(Instancias)
        objLineal = LinealEquation(x,y,w,b)
        grafica.set_title("Red Neuronal Lineal")
        grafica.scatter(
            x,
            y,
            label="Datos reales"
        )
        grafica.plot(
          x,
          objLineal.prueba(),#Me devuelve la ultima iteracion de la prediccion para posteriormente graficarla
          color='red',
          label="Predicción"
        )

    elif tipoFuncion == "FuncionCuadratica":
        w = np.random.randn()  # peso
        b = np.random.randn()  # bias de la red(direcciones)
        z = np.random.randn()
        x = np.linspace(-1,1,20)#Valores mejores ajustados para la funcion Cuadratica en este caso
        y = x**2#Funcion a aprender 
        grafica.plot(x, y)
        #Comunicaciones de Objetos(Instancias)
        objCuadratica = SquareEquation(x,y,w,b,z)
        grafica.set_title("Red Neuronal Cuadrática")
        grafica.scatter(
            x,
            y,
            label="Datos reales"
        )
        
        grafica.plot(
          x,
          objCuadratica.prueba(),#Me devuelve la ultima iteracion de la prediccion para posteriormente graficarla
          color='red',
          label="Predicción"
        )

    canvas.draw()#Dibuja Grafica


