import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# Crear el dataset

n = 500 # Número de registros
p = 2 # Número de características

X, Y = make_circles(n_samples=n, factor=0.5, noise=0.05)
Y = Y[:, np.newaxis] # Convertir Y en un vector columna

plt.scatter(X[Y[:,0]== 0, 0], X[Y[:,0] == 0, 1], c="skyblue")
plt.scatter(X[Y[:,0] == 1, 0], X[Y[:,0] == 1, 1], c="salmon")
plt.axis("equal")
 

# Clase de la capa de la red neuronal

class neural_layer: # Representa una capa de la red neuronal
    def __init__(self, n_conn, n_neur, act_f): # n_conn: número de conexiones con la capa anterior, n_neur: número de neuronas
        self.act_f = act_f
        self.b = np.random.rand(1, n_neur) * 2 - 1 # Bias para cada neurona, multiplicado por 2 y restado por 1 para que esté entre -1 y 1
        self.W = np.random.rand(n_conn, n_neur) * 2 - 1 # Pesos para cada conexión, multiplicado por 2 y restado por 1 para que esté entre -1 y 1


# Funciones de activación

sigm = (lambda x: 1 / (1 + np.e ** (-x)), # Función sigmoide
        lambda x: x * (1 - x)) # Derivada de la función sigmoide
         
relu = lambda x: np.maximum(0, x) # Función de activación ReLU
  

# Crear la red neuronal


def create_nn(topology, act_f):
    nn = [] # Red neuronal, lista de capas
    for i, layer in enumerate(topology[:-1]): # Recorrer todas las capas de la red neuronal, excepto la última para evitar overflow
       nn.append(neural_layer(topology[i], topology[i + 1], act_f))
    return nn

topology = [p, 4, 8, 16,8,4,1] # Número de neuronas en cada capa
neural_net = create_nn(topology, sigm)

# Función de coste, en este caso, el error cuadrático medio, junto con su derivada

l2_cost = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2), # Función de coste
           lambda Yp, Yr: (Yp - Yr)) # Derivada de la función de coste


# Función de entrenamiento

def train(neural_net, X, Y, l2_cost, lr=0.05, train=True):
    out = [(None, X)]  # Salida de cada capa, inicializada con las entradas

    # Forward pass
    for i, layer in enumerate(neural_net):
        z = out[-1][1] @ layer.W + layer.b
        a = layer.act_f[0](z)
        out.append((z, a))

    if not train:
        return out[-1][1]

    # Backward pass
    deltas = []
    for i in reversed(range(len(neural_net))):
        z = out[i + 1][0]
        a = out[i + 1][1]

        if i == len(neural_net) - 1:
            # Delta de la última capa
            delta = l2_cost[1](a, Y) * neural_net[i].act_f[1](a)
        else:
            # Delta de las capas ocultas
            delta = deltas[0] @ neural_net[i + 1].W.T * neural_net[i].act_f[1](a)
        deltas.insert(0, delta)

        # Actualización de pesos y bias
        neural_net[i].b -= np.mean(delta, axis=0, keepdims=True) * lr
        neural_net[i].W -= out[i][1].T @ delta * lr

    return out[-1][1]

    

# Entrenar la red neuronal

import time

neural_n = create_nn(topology, sigm)

loss = []

for i in range(25000):

    pY = train(neural_n, X, Y, l2_cost, lr=0.005)

    if i %500 == 0:
        
        loss.append(l2_cost[0](pY, Y))

        res = 50

        _x0 = np.linspace(-1.5, 1.5, res)
        
        _Y = np.zeros((res, res))

        for i0, x0 in enumerate(_x0):
            for i1, x1 in enumerate(_x0):
                _Y[i0, i1] = train(neural_n, np.array([[x0, x1]]), Y, l2_cost, train=False)[0][0]

        plt.pcolormesh(_x0, _x0, _Y, cmap="coolwarm")
        plt.axis("equal")

        plt.scatter(X[Y[:,0] == 0, 0], X[Y[:,0] == 0, 1], c="skyblue")
        plt.scatter(X[Y[:,0] == 1, 0], X[Y[:,0] == 1, 1], c="salmon")
        
        plt.show()
        plt.plot(range(len(loss)), loss)
        plt.show()
        time.sleep(0.5)

print("Entrenamiento finalizado")
print("Error final: ", loss[-1])