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
        z = act_func[i](z @ P[i])

    return z


def cost(x, P, act_func, y):
    y_pred = feedforward(x, P, act_func)
    return np.mean((y_pred - y)**2)


np.random.seed(42)


def y_func(x):
    return np.cos(x)


x = np.array([[i] for i in np.linspace(0, 2 * np.pi, 1000)])


y = y_func(x)

P = 2 * [None]

P[0] = np.random.normal(0, 1, (2, 20))
P[0][-1] = 0.01 * np.ones(20)

print(P[0])

P[1] = np.random.normal(0, 1, (21, 1))
P[1][-1] = 0.01 * np.ones(1)


act_func = [np.tanh, lambda x: x]

feedforward(x, P, act_func)

grad = elementwise_grad(cost, 1)

for i in range(10000):
    print(cost(x, P, act_func, y))
    gradient = grad(x[idx], P, act_func, y[idx])
    P[0] -= 0.01 * gradient[0]
    P[1] -= 0.01 * gradient[1]

plt.plot(x, feedforward(x, P, act_func))
plt.plot(x, y)
plt.show()
