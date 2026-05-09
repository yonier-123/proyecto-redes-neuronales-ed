import tkinter as tk
from tkinter import ttk#Para JComboBox
from guiGraphic import graphics#Para comunicarme con la clase graphics

def getValorForGraphic():
    tipoFuncion = combo.get()#Captura el tipo de funcion escojida
    graphics.actualizarGrafica(tipoFuncion)#Llama metodo que se encarga de actualizar Grafica
    combo.set("")

def getTexto():
    texto["text"]=entrada.get() #Entrada cambiante   
def iniciar():
    global texto,tipoFuncion#Defino que es una variable global
    global entrada
    global combo,canva
    global objGraphic,ventana
#Defino como vacio el canva(Grafica)
    canva=None
#Creo ventana
    ventana = tk.Tk()#nace la aplicacion  #ancho*alto
    ventana.geometry("800x600")#SetBounds pero en python
#Agrego texto dentro de la interfaz
    texto = tk.Label(ventana,text= "Red Neuronal")
    ventana.resizable(False,False)#Para que no se redimencione la pantalla

#Agrego boton dentro de la interfaz  (gui,buttonName,funcion/evento)
    botonTraining = tk.Button(ventana,text="EntrenarRedNeuronal",command=getValorForGraphic)#Captura Evento del JComboBox
    
#JTextField
    entrada = tk.Entry(ventana)
    
#ComboBox
    combo = ttk.Combobox(ventana,values=["FuncionLineal","FuncionCuadratica",
                                         "FuncionCubica"])
#Creacion del grafico dentro de la interfaz
    canva = graphics.crearGraphics(ventana)
    canva.get_tk_widget().pack()
    
    entrada.pack()#Es como el add(objeto) en java
    combo.pack()
    texto.pack()#Agrega el objeto dentro de la interfaz
    botonTraining.pack()

#loop infinito para que siempre se ejecute la ventana
    ventana.mainloop()

