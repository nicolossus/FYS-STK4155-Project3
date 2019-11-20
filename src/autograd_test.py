import autograd.numpy as np
from autograd import grad
from autograd import elementwise_grad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def feedforward(x, P, act_func):
    dim = len(P)
    z = x
    for i in range(dim):
        z = np.concatenate((z, np.ones((z.shape[0], 1))), axis=1)
        z = np.tanh(z @ P[i])

    return z


def cost(x, P, act_func, y):
    y_pred = feedforward(x, P, act_func)
    return np.mean((y_pred - y)**2)


np.random.seed(42)


def y_func(x):
    return np.sin(np.pi * x[:, 0]) * np.sin(2 * np.pi * x[:, 1]) - \
        np.sin(np.pi * x[:, 1]) * np.sin(2 * np.pi * x[:, 0])


x = np.random.uniform(0, 1, (1000, 2))

y = y_func(x)

P = 2 * [None]

P[0] = np.random.normal(0, 1, (3, 10))
P[0][-1] = 0.01 * np.ones(10)

P[1] = np.random.normal(0, 1, (11, 1))
P[1][-1] = 0.01 * np.ones(1)


act_func = [np.tanh, lambda x: x]

grad = elementwise_grad(cost, 1)

for i in range(100):
    print(cost(x, P, act_func, y))
    gradient = grad(x, P, act_func, y)
    P[0] -= 0.05 * gradient[0]
    P[1] -= 0.05 * gradient[1]
