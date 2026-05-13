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
    ventana.title("Simulador de Red Neuronal")
    ventana.configure(bg="#eef2f7")
    ventana.geometry("900x600")#SetBounds pero en python
    ventana.option_add("*Font", ("Segoe UI", 10))

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TCombobox", padding=4)
    style.configure("Accent.TButton", background="#2563eb", foreground="white", padding=(12, 6))
    style.map("Accent.TButton", background=[("active", "+#1d4ed8")])
#Agrego texto dentro de la interfaz
    header = tk.Label(
        ventana,
        text="Simulador Red Neuronal",
        bg="#eef2f7",
        fg="#0f172a",
        font=("Segoe UI", 18, "bold")
    )
    ventana.resizable(False,False)#Para que no se redimencione la pantalla

    controlFrame = tk.Frame(ventana, bg="#ffffff", highlightthickness=1, highlightbackground="#dbe4f0")
    textFrame = tk.Frame(ventana, bg="#ffffff", highlightthickness=1, highlightbackground="#dbe4f0")
    plotFrame = tk.Frame(ventana, bg="#ffffff", highlightthickness=1, highlightbackground="#dbe4f0")

    textEpoch = tk.Label(controlFrame,text="Veces que Quiero Entrenar La redNeuronal", bg="#ffffff", fg="#334155")
    textTipoFuncion = tk.Label(controlFrame,text="Selecciona Tipo de Funcion a Entrenar", bg="#ffffff", fg="#334155")
    textW=tk.Label(controlFrame,text="Ingresa valor W", bg="#ffffff", fg="#334155")
    textB=tk.Label(controlFrame,text="Ingresa valor B", bg="#ffffff", fg="#334155")
    textW2=tk.Label(controlFrame,text="Ingresa valor W2", bg="#ffffff", fg="#334155")

#Agrego boton dentro de la interfaz  (gui,buttonName,funcion/evento)
    botonTraining = tk.Button(
        controlFrame,
        text="Entrenar Red Neuronal",
        command=entrenar,
        bg="#2563eb",
        fg="white",
        activebackground="#1d4ed8",
        activeforeground="white",
        relief="flat",
        padx=10,
        pady=6
    )#Captura Evento del JComboBox

#JTextField  #Epoch quiere decir una pasada completa por todos los datos de entrenamiento
    entradaEpoch = tk.Entry(controlFrame, relief="flat", highlightthickness=1, highlightbackground="#cbd5e1")#Valor para saber cuantas iteraciones voy a hacer en el for
    entradaW = tk.Entry(controlFrame, relief="flat", highlightthickness=1, highlightbackground="#cbd5e1")
    entradaB = tk.Entry(controlFrame, relief="flat", highlightthickness=1, highlightbackground="#cbd5e1")
    entradaW2 = tk.Entry(controlFrame, relief="flat", highlightthickness=1, highlightbackground="#cbd5e1")

#JTextArea
    textArea = tk.Text(textFrame, width=40, height=15, bg="#f8fafc", fg="#0f172a", relief="flat", highlightthickness=1, highlightbackground="#cbd5e1")
    
#Scrollbar  para poder subir y bajar en el TextArea
    scroll = tk.Scrollbar(textFrame)#Lo agrego ala Interfaz
    scroll.config(command=textArea.yview)#Conectar el scroll con el TextArea
    textArea.config(yscrollcommand=scroll.set)#Conecto TextArea con Scrollbar
#ComboBox
    combo = ttk.Combobox(controlFrame,values=["FuncionLineal","FuncionCuadratica",
                                         "FuncionCubica","FuncionLogaritmica",
                                         "FuncionExponencial","FuncionTrigonometrica",
                                         "FuncionCircunferencia"])
#Creacion del grafico dentro de la interfaz
    canva = graphics.crearGraphics(plotFrame)
    canva.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)#El place reemplaza al pack(add(object) en java)
    #Coordenadas de donde estan ubicados los Componentes en la Interfaz Grafica
    header.place(x=20,y=12)
    plotFrame.place(x=20,y=50, width=480, height=350)
    controlFrame.place(x=520,y=20, width=350, height=260)
    textFrame.place(x=520,y=295, width=350, height=265)

    textTipoFuncion.place(x=20,y=20)
    combo.place(x=20,y=42, width=300)
    textEpoch.place(x=20,y=76)
    entradaEpoch.place(x=20,y=98, width=120)#Es como el setBound(coordenadas) en java
    textW.place(x=20,y=136)
    entradaW.place(x=20,y=158, width=90)
    textB.place(x=125,y=136)
    entradaB.place(x=125,y=158, width=90)
    textW2.place(x=230,y=136)
    entradaW2.place(x=230,y=158, width=90)
    botonTraining.place(x=90,y=190)

    textArea.pack(side="left", fill="both", expand=True, padx=(8,0), pady=8)
    scroll.pack(side="right", fill="y", pady=8, padx=(0,8))


#loop infinito para que siempre se ejecute la ventana
    ventana.mainloop()

