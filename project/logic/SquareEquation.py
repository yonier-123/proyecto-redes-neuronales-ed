import numpy as np
import matplotlib.pyplot as plt

class SquareEquation:
    def __init__(self,x,y,w,b,z):
        self.x=x
        self.y=y
        self.w=w
        self.b=b
        self.z=z
        self.textDetails=""

    #ax^2+bx+c
    def forward(self):#Prediccion Poner la red neuronal a pensar
        return self.w * self.x**2 + self.b*self.x+self.z#Dado un valor x la red neuronal devuelve lo que cree que es y
    #Ejemplo dado w=2,b=0,x=3
    # y_prediction = ax^2+bx+c  esa es la forma de la funcion

    #Calculo del error en este caso cuadratico
    def loss(self,y_predit, y): # Se usa para facilitar la optmizacion(derivaradas)
        return np.mean(( y_predit - y )**2)#Penaliza errores grandes de una Ej:(4)^2 = 16
    # np.mean se encarga de calcular el promedio del error

    #  dw=2mean((ypred-y)x^2)
    #  db=2⋅mean((ypred​−y)⋅x)
    #  dz=2⋅mean(ypred​−y)
    #Gradiente de la red neuronal
    def backward(self,x, y, y_predit):#Como debe cambiar w y b para reducir el error
        error = y_predit - y  #La gradiante le dice ala red neuronal por donde moverse basado en el calculo del error
        dw = 2 * np.mean(error* x**2)
        db = 2 * np.mean(error *x) #Error entreo lo que genera vs el valor real
        dz = 2 * np.mean(error)
        return dw,db,dz #Devuelve correccion

# learning rate para aprendisaje estable ej si se pone 1 aprenderia mal por lo rapido que va ir
# Actualiza parametros w y b para reducir el error
#Usa estas formulas : w = w - lr * dw  
#b = b - lr * db
    def actualizar(self,dw,db,dz):
   # global w, b#Variables fuera de la funcion 
        lr = 0.1
        w = self.w - lr * dw
        b = self.b - lr * db
        z = self.z - lr * dz
        return w,b,z 


    def prueba(self,n_epoch):
        self.textDetails = "" #Para refrescar Variable y se le puedan meter mas valores 
    #Prueba de 20 iteraciones repitiendo el proceso de
    # predecir ,de perdida, de calcular la gradiente ,de actualizar los valores de w y b para mejorar precision
    #Finalmente imprimir cada iteracion mostrando el proceso de aprendisaje
        for epoch in range(n_epoch):#Entre mas iteraciones mas precisa se vuelve la red neuronal
            y_predit = self.forward()  
                              #Con 20 todavia le falta
            l = self.loss(y_predit, self.y) #Con 30 es mucho mas precisa
                              #Con 40 mejor aun es muy exacta
            dw, db,dz = self.backward(self.x, self.y, y_predit)
            w,b,z = self.actualizar(dw, db,dz)
            self.w = w#Actualizacion de la variable
            self.b=b
            self.z=z
            self.textDetails+=f"Epoch {epoch}: loss={l:.4f}, w={w:.2f}, b={b:.2f}\n" #\n indica salto de linea
            print(f"Epoch {epoch}: loss={l:.4f}, w={w:.2f}, b={b:.2f}, z={z:.2f}")
        return self.forward()

    def visualizacion(self):
        #Metodos para graficar la red neuronal de esta funcion
            plt.scatter(self.x, self.y, label="Datos reales")
            plt.plot(self.x, self.forward(), color='red', label="Predicción")
            plt.legend()
            plt.show()#Muestra grafico final

    def getTextDetails(self):
        return self.textDetails