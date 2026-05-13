from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np#Para realizar calculos(Ayuda para Graficar)

#Importaciones para comunicarme con las clases de las redes neuronales programadas
from logic.LinealEquation import LinealEquation#Para comunicarme con la clase LinealEquation
from logic.SquareEquation import SquareEquation#Para comunicarme con la red que me genera una cuadratica
from logic.LogarithmicEquation import LogarithmicEquation
from logic.ExponentialEquation import ExponentialEquation
from logic.TrigonometricEquation import TrigonometricEquation
from logic.CircleEquation import CircleEquation


def crearGraphics(ventana):
    global objLineal
    global x,y,w,b,z
    global grafica,canvas

    #Crear Grafica   #(5,4)
    figura = Figure(figsize=(5,4),dpi=100)

    grafica = figura.add_subplot(111)
    x = np.array([-1, -0.5, 0, 0.5, 1])#Datos de entrada red neuronal
    #y = y = 2 * x +5
    #grafica.plot(x,y)

   

    grafica.legend()
    canvas = FigureCanvasTkAgg(figura,master=ventana)
    canvas.draw()
    return canvas

def capturarTextDetails():
        return textDetails

def actualizarGrafica(tipoFuncion,epoch,w1,b1,w2):
    global textDetails
    grafica.clear()#Limpia Grafica para poner otra 
   
    x = np.array([-1,-0.5,0,0.5,1])#Valores para la red neuronal en x

    if tipoFuncion == "FuncionLineal":
        w = np.random.randn()  # peso
        b = np.random.randn()  # bias de la red(direcciones)
        #y = 2*x + 5
        y=w1*x + b1
        grafica.plot(x, y)
        #Comunicaciones de Objetos(Instancias)
        objLineal = LinealEquation(x,y,w,b)
        grafica.set_title("Red Neuronal F.Lineal y = wx+b")
        grafica.scatter(
            x,
            y,
            label="Datos reales"
        )
        grafica.plot(
          x,
          objLineal.prueba(epoch),#Me devuelve la ultima iteracion de la prediccion para posteriormente graficarla
          color='red',
          label="Predicción"
        )
        textDetails = objLineal.getTextDetails()
        capturarTextDetails()


    elif tipoFuncion == "FuncionCuadratica":
        w = np.random.randn()  # peso
        b = np.random.randn()  # bias de la red(direcciones)
        z = np.random.randn()
        x = np.linspace(-1,1,20)#Valores mejores ajustados para la funcion Cuadratica en este caso
       # y = x**2#Funcion a aprender
        y = w1*x**2+w2*x+b1 
        grafica.plot(x, y)
        #Comunicaciones de Objetos(Instancias)
        objCuadratica = SquareEquation(x,y,w,b,z)
        grafica.set_title("Red Neuronal F.Cuadrática ")
        grafica.scatter(
            x,
            y,
            label="Datos reales"
        )
        
        grafica.plot(
          x,
          objCuadratica.prueba(epoch),#Me devuelve la ultima iteracion de la prediccion para posteriormente graficarla
          color='red',
          label="Predicción"
        )
        
        textDetails = objCuadratica.getTextDetails()
        capturarTextDetails()

    elif tipoFuncion == "FuncionLogaritmica":
        w = np.random.randn()
        b = np.random.randn()
        x = np.linspace(0.1, 2, 20)  # x debe ser positivo para ln(x)
        y = w1 * np.log(x) + b1
        grafica.plot(x, y)
        objLog = LogarithmicEquation(x, y, w, b)
        grafica.set_title("Red Neuronal F.Logarítmica y = w·ln(x)+b")
        grafica.scatter(x, y, label="Datos reales")
        grafica.plot(x, objLog.prueba(epoch), color='red', label="Predicción")
        textDetails = objLog.getTextDetails()
        capturarTextDetails()

    elif tipoFuncion == "FuncionExponencial":
        if w2 == 0:
            w2 = 1  # evitar línea horizontal degenerada
        w = np.random.randn()
        b = np.random.randn()
        z = np.random.randn()
        x = np.linspace(-1, 1, 20)
        y = w1 * np.exp(w2 * x) + b1
        grafica.plot(x, y)
        objExp = ExponentialEquation(x, y, w, b, z)
        grafica.set_title("Red Neuronal F.Exponencial y = w·e^(z·x)+b")
        grafica.scatter(x, y, label="Datos reales")
        grafica.plot(x, objExp.prueba(epoch), color='red', label="Predicción")
        textDetails = objExp.getTextDetails()
        capturarTextDetails()

    elif tipoFuncion == "FuncionTrigonometrica":
        if w2 == 0:
            w2 = 1  # evitar línea horizontal degenerada
        w = np.random.randn()
        b = np.random.randn()
        z = np.random.randn()
        x = np.linspace(-2, 2, 20)
        y = w1 * np.sin(w2 * x) + b1
        grafica.plot(x, y)
        objTrig = TrigonometricEquation(x, y, w, b, z)
        grafica.set_title("Red Neuronal F.Trigonométrica y = w·sin(z·x)+b")
        grafica.scatter(x, y, label="Datos reales")
        grafica.plot(x, objTrig.prueba(epoch), color='red', label="Predicción")
        textDetails = objTrig.getTextDetails()
        capturarTextDetails()

    elif tipoFuncion == "FuncionCircunferencia":
        w = np.random.randn() * 0.5 + 1.0  # radio inicial positivo
        b = np.random.randn()
        z = np.random.randn()
        t = np.linspace(0, 2 * np.pi, 20)
        x_real = w2 + w1 * np.cos(t)
        y_real = b1 + w1 * np.sin(t)
        objCirc = CircleEquation(x_real, y_real, w, b, z)
        grafica.set_title("Red Neuronal F.Circunferencia")
        grafica.scatter(x_real, y_real, label="Datos reales")
        x_plot, y_plot = objCirc.prueba(epoch)
        grafica.plot(x_plot, y_plot, color='red', label="Predicción")
        grafica.set_aspect('equal', adjustable='box')
        textDetails = objCirc.getTextDetails()
        capturarTextDetails()

    elif tipoFuncion=="":
        textDetails=""#Queda Vacio porque no se ha seleccionado una red neuronal que aprende una funcion
        capturarTextDetails()

    canvas.draw()#Dibuja Grafica


