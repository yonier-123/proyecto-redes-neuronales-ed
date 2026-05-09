import numpy as np
import matplotlib.pyplot as plt

class LinealEquation:
    def __init__(self,x,y,w,b):
        self.x=x
        self.y=y
        self.w=w
        self.b=b
        

    def forward(self):#Prediccion Poner la red neuronal a pensar
        return self.w * self.x + self.b#Dado un valor x la red neuronal devuelve lo que cree que es y
    #Ejemplo dado w=2,b=0,x=3
    # y_prediction = (2)(3)+ 0 = 6  la red dice para x creo que y=6 

    #Calculo del error en este caso cuadratico
    def loss(self,y_predit, y): # Se usa para facilitar la optmizacion(derivaradas)
        return np.mean(( y_predit - y )**2)#Penaliza errores grandes de una Ej:(4)^2 = 16
    # np.mean se encarga de calcular el promedio del error


    #  dw=2⋅mean((ypred​−y)⋅x)
    #  db=2⋅mean(ypred​−y)
    #Gradiente de la red neuronal
    def backward(self,x, y, y_predit):#Como debe cambiar w y b para reducir el error
        error = y_predit - y  #La gradiante le dice ala red neuronal por donde moverse basado en el calculo del error
        dw = 2 * np.mean(error *x) #Error entreo lo que genera vs el valor real
        db = 2 * np.mean(error)
        return dw,db #Devuelve correccion

# learning rate para aprendisaje estable ej si se pone 1 aprenderia mal por lo rapido que va ir
# Actualiza parametros w y b para reducir el error
#Usa estas formulas : w = w - lr * dw  
#b = b - lr * db
    def actualizar(self,dw,db):
   # global w, b#Variables fuera de la funcion 
        lr = 0.1
        w = self.w - lr * dw
        b = self.b - lr * db
        return w,b 


    def prueba(self):
    #Prueba de 20 iteraciones repitiendo el proceso de
    # predecir ,de perdida, de calcular la gradiente ,de actualizar los valores de w y b para mejorar precision
    #Finalmente imprimir cada iteracion mostrando el proceso de aprendisaje
        for epoch in range(30):#Entre mas iteraciones mas precisa se vuelve la red neuronal
            y_predit = self.forward()  
                              #Con 20 todavia le falta
            l = self.loss(y_predit, self.y) #Con 30 es mucho mas precisa
                              #Con 40 mejor aun es muy exacta
            dw, db = self.backward(self.x, self.y, y_predit)
            w,b = self.actualizar(dw, db)
            self.w = w#Actualizacion de la variable
            self.b=b
            print(f"Epoch {epoch}: loss={l:.4f}, w={w:.2f}, b={b:.2f}")
        return self.forward()

    def visualizacion(self):
        #Metodos para graficar la red neuronal de esta funcion
            plt.scatter(self.x, self.y, label="Datos reales")
            plt.plot(self.x, self.forward(), color='red', label="Predicción")
            plt.legend()
            plt.show()#Muestra grafico final
