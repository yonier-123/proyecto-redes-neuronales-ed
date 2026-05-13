import numpy as np
import matplotlib.pyplot as plt

class CircleEquation:
    def __init__(self, x, y, w, b, z):
        # x, y son arrays con coordenadas de puntos en la circunferencia
        # w = radio (r), b = centro_y (cy), z = centro_x (cx)
        self.x = x
        self.y = y
        self.w = w
        self.b = b
        self.z = z
        self.textDetails = ""

    def forward(self):
        # Error implicito de la ecuacion del circulo: (x-cx)^2 + (y-cy)^2 - r^2
        return (self.x - self.z) ** 2 + (self.y - self.b) ** 2 - self.w ** 2

    def loss(self, y_pred, y):
        # En este modelo no usamos y_pred tradicional; usamos el error implicito
        error = self.forward()
        return np.mean(error ** 2)

    def backward(self):
        error = self.forward()
        # Derivadas parciales del loss = mean(error^2)
        # d/dcx = 2*mean(error * 2*(cx - x)) = 4*mean(error*(cx - x))
        # d/dcy = 4*mean(error*(cy - y))
        # d/dr  = 2*mean(error * (-2r)) = -4*r*mean(error)
        da = 4 * np.mean(error * (self.z - self.x))
        db = 4 * np.mean(error * (self.b - self.y))
        dr = -4 * self.w * np.mean(error)
        return da, db, dr

    def actualizar(self, da, db, dr):
        lr = 0.01
        a = self.z - lr * da
        b = self.b - lr * db
        r = self.w - lr * dr
        # El radio no puede ser negativo ni cero
        if r < 0.01:
            r = 0.01
        return a, b, r

    def prueba(self, n_epoch):
        self.textDetails = ""
        for epoch in range(n_epoch):
            error = self.forward()
            l = np.mean(error ** 2)
            da, db, dr = self.backward()
            a, b, r = self.actualizar(da, db, dr)
            self.z = a
            self.b = b
            self.w = r
            self.textDetails += f"Epoch {epoch}: loss={l:.4f}, cx={a:.2f}, cy={b:.2f}, r={r:.2f}\n"
            print(f"Epoch {epoch}: loss={l:.4f}, cx={a:.2f}, cy={b:.2f}, r={r:.2f}")
        # Retorna puntos parametricos para graficar el circulo completo
        t = np.linspace(0, 2 * np.pi, 100)
        x_plot = self.z + self.w * np.cos(t)
        y_plot = self.b + self.w * np.sin(t)
        return x_plot, y_plot

    def visualizacion(self):
        plt.scatter(self.x, self.y, label="Datos reales")
        x_plot, y_plot = self.prueba(30)
        plt.plot(x_plot, y_plot, color='red', label="Predicción")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.show()

    def getTextDetails(self):
        return self.textDetails
