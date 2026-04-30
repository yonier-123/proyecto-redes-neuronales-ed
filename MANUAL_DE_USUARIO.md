# Manual de Usuario — Neural ODE Simulator

Bienvenido al simulador interactivo de Redes Neuronales y Ecuaciones Diferenciales (Neural ODEs). Este documento explica en detalle cada uno de los paneles, gráficas y controles que ves en la interfaz, cómo interpretarlos y qué representan matemáticamente.

---

## 🧭 Estructura General de la Interfaz

La aplicación está dividida en tres columnas o paneles principales:

1. **Panel Izquierdo:** La arquitectura de la Red Neuronal y los controles interactivos.
2. **Panel Central:** El comportamiento físico/dinámico del sistema (Trayectoria y Campo Vectorial).
3. **Panel Derecho:** El proceso de "Aprendizaje" o Entrenamiento (Superficie de Error y Convergencia).

---

## 1. Panel Izquierdo: Red Neuronal y Controles

### 🧠 Grafo de la Red Neuronal (Arriba Izquierda)
Muestra la arquitectura interna de la red neuronal que está controlando el sistema.
- **Nodos de Entrada (x, y):** Representan la posición actual de un punto en el espacio 2D.
- **Nodos de Salida (dx/dt, dy/dt):** Representan la "velocidad" o hacia dónde debe moverse el punto en el siguiente instante de tiempo.
- **Líneas (Conexiones):** Son los pesos matemáticos (`w1, w2, w3, w4`) de la red. 
  - **Color Azul:** Peso positivo (aumenta el valor).
  - **Color Rojo:** Peso negativo (disminuye el valor).
  - **Grosor:** Entre más gruesa es la línea, mayor es la influencia (magnitud) de esa conexión. Al entrenar la red, verás estas líneas palpitar y cambiar de color a medida que la red "aprende".

### 🎛️ Controles y Parámetros (Abajo Izquierda)
- **PLAY / PAUSE / RESET:** Controlan el ciclo de entrenamiento. *Play* inicia el descenso de gradiente (la red empieza a aprender), *Pause* lo detiene para que observes, y *Reset* borra el progreso y mezcla los pesos al azar.
- **Preset:** Elige la figura geométrica o trayectoria que quieres que la red intente imitar (Ej. un círculo perfecto, o una espiral).
- **Optimizador:** El algoritmo matemático que busca el valle de error. 
  - *SGD:* Básico, un poco lento.
  - *Momentum:* Gana inercia, ideal para no estancarse.
  - *Adam:* El más avanzado, se adapta inteligentemente y suele converger más rápido.
- **Solver:** El método numérico para resolver la Ecuación Diferencial. *Euler* es sencillo pero puede volverse inestable; *RK4* (Runge-Kutta 4) es muy preciso y estable.
- **Learning Rate (Tasa de aprendizaje):** Qué tan "grandes" son los pasos que da la red al aprender. Si es muy bajo, aprende muy lento. Si es muy alto, el sistema "explota" o diverge.
- **Velocidad:** Controla qué tan rápido se actualizan los gráficos visualmente en tu pantalla (de 1x a 10x).
- **Pesos manuales (Sliders w1...b2):** Te permiten mover los valores matemáticos de la red a mano para ver cómo afectan el sistema en tiempo real. *(Nota: solo funcionan cuando el entrenamiento está en PAUSE).*
- **Boton EXPORT PNG:** Toma una "fotografía" de alta calidad de todos los paneles y los guarda en tu computador.

---

## 2. Panel Central: Dinámica del Sistema

### 🌪️ Trayectoria 2D y Campo Vectorial (Arriba Centro)
Este es el "Espacio de Fase" o el mundo donde vive nuestra partícula.
- **Línea Verde Punteada:** Es el **objetivo**. Es la ruta que queremos que la red neuronal aprenda a trazar.
- **Línea Sólida que cambia de color:** Es la trayectoria que la red neuronal está generando **actualmente**. Su color se degrada de púrpura (inicio del tiempo) a amarillo (final del tiempo). Al presionar *PLAY*, verás cómo esta línea intenta "doblarse" hasta encajar perfectamente sobre la línea verde.
- **Fondo con flechas grises (Campo Vectorial):** Muestra el "viento" o las "corrientes" en todo el espacio 2D. Es decir, si colocas una partícula en cualquier punto del plano, las flechas indican hacia dónde la empujaría la red neuronal.

### 🌡️ Heatmaps de Activaciones (Abajo Centro)
Muestra una "radiografía" térmica de las dos ecuaciones individuales de la red.
- **Panel izquierdo (dx/dt):** Muestra la velocidad horizontal.
- **Panel derecho (dy/dt):** Muestra la velocidad vertical.
- **Interpretación:** El color **Rojo** significa velocidad negativa (empuja hacia la izquierda o hacia abajo). El color **Azul** significa velocidad positiva (empuja hacia la derecha o hacia arriba). La línea blanca central es donde la velocidad es exactamente cero.

---

## 3. Panel Derecho: El Proceso de Aprendizaje

### 🏔️ Paisaje de Pérdida 3D / Loss Landscape (Arriba Derecha)
Muestra el "terreno" matemático por el que la red está navegando para aprender.
- **La superficie 3D:** Representa el Error (Loss). Las montañas amarillas son zonas donde la red se equivoca mucho. Los valles morados profundos son zonas donde la red acierta (la trayectoria generada es idéntica a la deseada).
- **La bola roja:** Eres tú (el estado actual de la red). 
- **Interpretación:** Al darle *PLAY*, verás cómo la bola roja "rueda" montaña abajo buscando el fondo del valle. Este rodamiento es literalmente la visualización del concepto matemático de **Descenso de Gradiente**. *(Nota: Por simplicidad visual, este gráfico 3D solo mapea 2 de los 6 pesos de la red: w1 y w2).*

### 📈 Curvas de Convergencia (Abajo Derecha)
Muestran las estadísticas vitales del cerebro de la red en 4 pequeños gráficos:
1. **Pérdida (MSE):** Mide el error total. **Debe tender a cero**. Si sube, algo anda mal (probablemente el *Learning Rate* está muy alto).
2. **Gradientes (∇L):** Mide "qué tan inclinada" está la montaña de error en la posición actual. Cuando la red llega al fondo del valle y no puede aprender más, el gradiente llega a cero.
3. **Evolución de Pesos:** Muestra cómo van cambiando los valores internos (`w1, w2...`) de la red a lo largo del tiempo.
4. **Tasa de Cambio (||ΔW||):** Muestra el tamaño de los pasos que la red está dando. Al inicio son pasos grandes, pero a medida que se acerca al mínimo (al fondo del valle), los pasos se vuelven más pequeñitos y precisos.

---

## 💡 ¿Cómo usarlo para entender conceptos?

1. **El efecto del Optimizador:** Pon el preset `espiral_entrada`. Entrena usando `SGD`. Verás que la bola roja baja lento. Ahora reinicia (`RESET`), cambia a `Adam` y dale PLAY. Verás que la bola roja baja rapidísimo y con una trayectoria mucho más inteligente.
2. **El peligro del Learning Rate alto:** Pon un Learning Rate alto (ej. `0.1`), elige SGD y dale PLAY. Verás que la bola roja en el gráfico 3D empieza a saltar violentamente de un lado a otro del valle sin poder quedarse en el fondo, e incluso el error puede explotar (irse al infinito).
3. **Exploración Manual:** Pon la aplicación en PAUSE. Empieza a mover los "Pesos manuales" de a poco. Mira cómo las flechas del "Campo Vectorial" en el panel central cambian de dirección, y cómo eso afecta drásticamente la ruta que toma la partícula. ¡Estás programando un sistema dinámico a mano!
