import numpy as np
import matplotlib.pyplot as plt


class RadicalEquation:
	def __init__(self, x, y, w, b):
		self.x = x
		self.y = y
		self.w = w
		self.b = b
		self.textDetails = ""

	# y = w * sqrt(x) + b
	def forward(self):
		return self.w * np.sqrt(np.abs(self.x)) + self.b

	def loss(self, y_pred, y):
		return np.mean((y_pred - y) ** 2)

	def backward(self, x, y, y_pred):
		error = y_pred - y
		dw = 2 * np.mean(error * np.sqrt(np.abs(x)))
		db = 2 * np.mean(error)
		return dw, db

	def actualizar(self, dw, db):
		lr = 0.1
		w = self.w - lr * dw
		b = self.b - lr * db
		return w, b

	def prueba(self, n_epoch):
		self.textDetails = ""
		for epoch in range(n_epoch):
			y_pred = self.forward()
			l = self.loss(y_pred, self.y)
			dw, db = self.backward(self.x, self.y, y_pred)
			w, b = self.actualizar(dw, db)
			self.w = w
			self.b = b
			self.textDetails += f"Epoch {epoch}: loss={l:.4f}, w={w:.2f}, b={b:.2f}\n"
			print(f"Epoch {epoch}: loss={l:.4f}, w={w:.2f}, b={b:.2f}")
		return self.forward()

	def visualizacion(self):
		plt.scatter(self.x, self.y, label="Datos reales")
		plt.plot(self.x, self.forward(), color='red', label="Predicción")
		plt.legend()
		plt.show()

	def getTextDetails(self):
		return self.textDetails
