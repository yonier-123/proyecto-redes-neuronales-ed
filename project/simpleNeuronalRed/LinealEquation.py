import numpy as np
import matplotlib.pyplot as plt

def forward(x):#Prediccion Poner la red neuronal a pensar
    return w * x + b#Dado un valor x la red neuronal devuelve lo que cree que es y
#Ejemplo dado w=2,b=0,x=3
# y_prediction = (2)(3)+ 0 = 6  la red dice para x creo que y=6 

#Calculo del error en este caso cuadratico
def loss(y_pred, y_real): # Se usa para facilitar la optmizacion(derivaradas)
    return np.mean((y_pred - y_real)**2)#Penaliza errores grandes de una Ej:(4)^2 = 16
# np.mean se encarga de calcular el promedio del error


 #  dw=2⋅mean((ypred​−y)⋅x)
 #  db=2⋅mean(ypred​−y)
#Gradiente de la red neuronal
def backward(x, y, y_pred):#Como debe cambiar w y b para reducir el error
    error = y_pred - y  #La gradiante le dice ala red neuronal por donde moverse basado en el calculo del error
    dw = 2 * np.mean(error * x) #Error entreo lo que genera vs el valor real
    db = 2 * np.mean(error)
    return dw, db#Devuelve correccion

lr = 0.1  # learning rate para aprendisaje estable ej si se pone 1 aprenderia mal por lo rapido que va ir
# Actualiza parametros w y b para reducir el error
#Usa estas formulas : w = w - lr * dw  
#b = b - lr * db
def actualizar(dw, db):
    global w, b#Variables fuera de la funcion 
    w = w - lr * dw
    b = b - lr * db


if __name__ == "__main__":
# Datos para red que aprende usando ypred = wx + b
    x = np.array([-1, -0.5, 0, 0.5, 1])#Datod de entrada red neuronal
    y = 2 * x  # lo que queremos que aprenda
    print(y)# -2,-1,0,1,2 Quedan Puntos en el Plano: (-1,-2) (-0.5,-1)
                 #(0,0) (0,5,1) (1,2)
    
  #Metodo random.randn() genera valores 0 <= numero < 1 tipo 0.1,0.3
    w = np.random.randn()  # peso
    b = np.random.randn()  # bias de la red(direcciones)
    #No se ponen valores como 0,1,2 porque la red neuronal no aprenderia correctamente
    
    
    #Prueba de 20 iteraciones repitiendo el proceso de
    # predecir ,de perdida, de calcular la gradiente ,de actualizar los valores de w y b para mejorar precision
    #Finalmente imprimir cada iteracion mostrando el proceso de aprendisaje
    for epoch in range(30):#Entre mas iteraciones mas precisa se vuelve la red neuronal
        y_predit = forward(x)  
                              #Con 20 todavia le falta
        l = loss(y_predit, y) #Con 30 es mucho mas precisa
                              #Con 40 mejor aun es muy exacta
        dw, db = backward(x, y, y_predit)
        actualizar(dw, db)
    
        print(f"Epoch {epoch}: loss={l:.4f}, w={w:.2f}, b={b:.2f}")
    
    #Metodos para graficar la red neuronal de esta funcion
    plt.scatter(x, y, label="Datos reales")
    plt.plot(x, forward(x), color='red', label="Predicción")
    plt.legend()
    plt.show()#Muestra grafico final
