import numpy as np
import matplotlib.pyplot as plt

class TrigonometricEquation:
    def __init__(self, x, y, w, b, z):
        # Forma: y = w * sin(z * x) + b
        self.x = x
        self.y = y
        self.w = w
        self.b = b
        self.z = z
        self.textDetails = ""

    def forward(self):
        return self.w * np.sin(self.z * self.x) + self.b

    def loss(self, y_pred, y):
        return np.mean((y_pred - y) ** 2)

    def backward(self, x, y, y_pred):
        error = y_pred - y
        sin_term = np.sin(self.z * x)
        cos_term = np.cos(self.z * x)
        dw = 2 * np.mean(error * sin_term)
        dz = 2 * np.mean(error * self.w * x * cos_term)
        db = 2 * np.mean(error)
        return dw, db, dz

    def actualizar(self, dw, db, dz):
        lr = 0.05
        w = self.w - lr * dw
        b = self.b - lr * db
        z = self.z - lr * dz
        return w, b, z

    def prueba(self, n_epoch):
        self.textDetails = ""
        for epoch in range(n_epoch):
            y_pred = self.forward()
            l = self.loss(y_pred, self.y)
            dw, db, dz = self.backward(self.x, self.y, y_pred)
            w, b, z = self.actualizar(dw, db, dz)
            self.w = w
            self.b = b
            self.z = z
            self.textDetails += f"Epoch {epoch}: loss={l:.4f}, w={w:.2f}, b={b:.2f}, z={z:.2f}\n"
            print(f"Epoch {epoch}: loss={l:.4f}, w={w:.2f}, b={b:.2f}, z={z:.2f}")
        return self.forward()

    def visualizacion(self):
        plt.scatter(self.x, self.y, label="Datos reales")
        plt.plot(self.x, self.forward(), color='red', label="Predicción")
        plt.legend()
        plt.show()

    def getTextDetails(self):
        return self.textDetails
