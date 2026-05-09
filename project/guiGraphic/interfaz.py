import tkinter as tk
from tkinter import ttk#Para JComboBox
from guiGraphic import graphics#Para comunicarme con la clase graphics

def entrenar():
    tipoFuncion = combo.get()#Captura el tipo de funcion escojida
    #Condicional que dice si la entrada de datos no es vacia que ejecute el metodo de actualizar epoch de lo contrario no
    if entradaEpoch.get()!="":
        getTexto(tipoFuncion)
    else:
        graphics.actualizarGrafica(tipoFuncion,30)#Llama metodo que se encarga de actualizar Grafica    
        inputDatesTextArea(graphics.capturarTextDetails())#Agrega Detalles del Entrenamiento
    combo.set("")

def getTexto(tipoFuncion):
    epoch=int(entradaEpoch.get()) #Entrada cambiante 
    graphics.actualizarGrafica(tipoFuncion,epoch)
    inputDatesTextArea(graphics.capturarTextDetails())#Agrega Detalles del Entrenamiento

def inputDatesTextArea(textDetails):
    textArea.delete(
        "1.0",
        tk.END
    )
    textArea.insert(
        tk.END,
        textDetails
    )
    textArea.see(tk.END)

def iniciar():
    global texto#Defino que es una variable global
    global entradaEpoch
    global textArea
    global combo,canva
    global objGraphic,ventana
#Defino como vacio el canva(Grafica)
    canva=None
#Creo ventana
    ventana = tk.Tk()#nace la aplicacion  #ancho*alto
    ventana.geometry("900x600")#SetBounds pero en python
#Agrego texto dentro de la interfaz
    textEpoch = tk.Label(ventana,text="Veces que Quiero Entrenar La redNeuronal")
    textTipoFuncion = tk.Label(ventana,text="Selecciona Tipo de Funcion a Entrenar")
    texto = tk.Label(ventana,text= "Simulador Red Neuronal")
    ventana.resizable(False,False)#Para que no se redimencione la pantalla

#Agrego boton dentro de la interfaz  (gui,buttonName,funcion/evento)
    botonTraining = tk.Button(ventana,text="EntrenarRedNeuronal",command=entrenar)#Captura Evento del JComboBox
    
#JTextField  #Epoch quiere decir una pasada completa por todos los datos de entrenamiento
    entradaEpoch = tk.Entry(ventana)#Valor para saber cuantas iteraciones voy a hacer en el for

#JTextArea
    textArea = tk.Text(ventana, width=40, height=15)
    
#Scrollbar  para poder subir y bajar en el TextArea
    scroll = tk.Scrollbar(ventana)#Lo agrego ala Interfaz
    scroll.config(command=textArea.yview)#Conectar el scroll con el TextArea
    textArea.config(yscrollcommand=scroll.set)#Conecto TextArea con Scrollbar
#ComboBox
    combo = ttk.Combobox(ventana,values=["FuncionLineal","FuncionCuadratica",
                                         "FuncionCubica"])
#Creacion del grafico dentro de la interfaz
    canva = graphics.crearGraphics(ventana)
    canva.get_tk_widget().place(x=20,y=20)#El place reemplaza al pack(add(object) en java)
    #Coordenadas de donde estan ubicados los Componentes en la Interfaz Grafica
    textEpoch.place(x=550,y=100) 
    entradaEpoch.place(x=550,y=120)#Es como el setBound(coordenadas) en java
    textTipoFuncion.place(x=550,y=30)
    combo.place(x=550,y=50)
    texto.place(x=440,y=0)#Agrega el objeto dentro de la interfaz
    botonTraining.place(x=550,y=180)
    textArea.place(x=550,y=250)
    scroll.place(x=865,y=250,height=245)#Ubicacion del Scroll dentro del TextArea

#loop infinito para que siempre se ejecute la ventana
    ventana.mainloop()

