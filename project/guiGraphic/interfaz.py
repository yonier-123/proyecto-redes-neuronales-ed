import tkinter as tk
from tkinter import ttk#Para JComboBox
from guiGraphic import graphics#Para comunicarme con la clase graphics

def entrenar():
    w=1#Defino variables para evitar errores
    b=0
    w2=1#Default 1 para que funciones exponencial/trigonométrica no sean horizontales
    tipoFuncion = combo.get()#Captura el tipo de funcion escojida
    #Capturar valor que va a tener las funciones
    if entradaW.get() !="":#Para Evitar que se dañe el programa en caso de no escribir en una casilla
       w = int(entradaW.get())
    if entradaB.get() != "":
        b = int(entradaB.get())
    if entradaW2.get() != "":
        w2 = int(entradaW2.get())
    
    #Condicional que dice si la entrada de datos no es vacia que ejecute el metodo de actualizar epoch de lo contrario no
    if entradaEpoch.get()!="":
        getTexto(tipoFuncion,w,b,w2)
    else:
        graphics.actualizarGrafica(tipoFuncion,30,1,0,1)#Llama metodo que se encarga de actualizar Grafica    
        inputDatesTextArea(graphics.capturarTextDetails())#Agrega Detalles del Entrenamiento
    combo.set("")


def getTexto(tipoFuncion,w,b,w2):
    epoch=int(entradaEpoch.get()) #Entrada cambiante 
    graphics.actualizarGrafica(tipoFuncion,epoch,w,b,w2)
    inputDatesTextArea(graphics.capturarTextDetails())#Agrega Detalles del Entrenamiento

def habilitacionInputs(event):
    kindOfFunction =combo.get()
    if kindOfFunction == "FuncionLineal":
        entradaW.config(state="normal")
        entradaB.config(state="normal")
        entradaW2.config(state="disabled")

    elif kindOfFunction == "FuncionCuadratica":
        entradaW.config(state="normal")
        entradaB.config(state="normal")
        entradaW2.config(state="normal")
    elif kindOfFunction == "FuncionLogaritmica":
        entradaW.config(state="normal")
        entradaB.config(state="normal")
        entradaW2.config(state="disabled")
    elif kindOfFunction == "FuncionExponencial":
        entradaW.config(state="normal")
        entradaB.config(state="normal")
        entradaW2.config(state="normal")
    elif kindOfFunction == "FuncionTrigonometrica":
        entradaW.config(state="normal")
        entradaB.config(state="normal")
        entradaW2.config(state="disabled")
    elif kindOfFunction == "FuncionCircunferencia":
        entradaW.config(state="normal")
        entradaB.config(state="normal")
        entradaW2.config(state="normal")

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
    global entradaEpoch,entradaW,entradaB,entradaW2
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
    textW=tk.Label(ventana,text="Ingresa valor W")
    textB=tk.Label(ventana,text="Ingresa valor B")
    textW2=tk.Label(ventana,text="Ingresa valor W2")
    texto = tk.Label(ventana,text= "Simulador Red Neuronal")
    ventana.resizable(False,False)#Para que no se redimencione la pantalla

#Agrego boton dentro de la interfaz  (gui,buttonName,funcion/evento)
    botonTraining = tk.Button(ventana,text="EntrenarRedNeuronal",command=entrenar)#Captura Evento del JComboBox

#JTextField  #Epoch quiere decir una pasada completa por todos los datos de entrenamiento
    entradaEpoch = tk.Entry(ventana)#Valor para saber cuantas iteraciones voy a hacer en el for
    entradaW = tk.Entry(ventana)
    entradaB = tk.Entry(ventana)
    entradaW2 = tk.Entry(ventana)

#JTextArea
    textArea = tk.Text(ventana, width=40, height=15)
    
#Scrollbar  para poder subir y bajar en el TextArea
    scroll = tk.Scrollbar(ventana)#Lo agrego ala Interfaz
    scroll.config(command=textArea.yview)#Conectar el scroll con el TextArea
    textArea.config(yscrollcommand=scroll.set)#Conecto TextArea con Scrollbar

    scrollHorizontal = tk.Scrollbar(ventana,orient="horizontal")
    scrollHorizontal.config(command=textArea.xview)
    textArea.config(xscrollcommand=scrollHorizontal.set)

#ComboBox
    combo = ttk.Combobox(ventana,values=["FuncionLineal","FuncionCuadratica",
                                         "FuncionCubica","FuncionLogaritmica",
                                         "FuncionExponencial","FuncionTrigonometrica",
                                         "FuncionCircunferencia"])
    combo.bind("<<ComboboxSelected>>",habilitacionInputs)#Para manejar Evento con ComboBox
# Habilitacion de Entradas de usuario
    entradaW.config(state="disabled")
    entradaB.config(state="disabled")
    entradaW2.config(state="disabled")


>>>>>>>>> Temporary merge branch 2
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
    entradaW.place(x=20,y=460)
    entradaB.place(x=20,y=510)
    entradaW2.place(x=200,y=460)
    textW.place(x=20,y=440)
    textB.place(x=20,y=490)
    textW2.place(x=200,y=440)
    scrollHorizontal.place(x=550,y=495,width=315)


#loop infinito para que siempre se ejecute la ventana
    ventana.mainloop()

