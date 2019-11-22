import autograd.numpy as np
from autograd import grad
from autograd import elementwise_grad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn import datasets


def feedforward(x, P, act_func):
    dim = len(P)
    z = x

    for i in range(dim):
        z = np.concatenate((z, np.ones((z.shape[0], 1))), axis=1)
        z = act_func[i](z @ P[i])

    return z


def g_trial(x, P, act_func):
    return x * (x - 6) * feedforward(x, P, act_func)


ddg_trial = elementwise_grad(elementwise_grad(g_trial, 0), 0)


def cost(x, P, act_func):
    RHS = ddg_trial(x, P, act_func)
    LHS = -g_trial(x, P, act_func)
    return np.mean((RHS - LHS)**2)


def generate_P(dim):
    n = len(dim) - 1
    P = n * [None]
    for i in range(n):
        P[i] = np.random.normal(0, 1, (dim[i] + 1, dim[i + 1]))
        P[i][-1] = 0.01 * np.ones(dim[i + 1])

    return P


np.random.seed(42)
x = np.array([[i] for i in np.linspace(0, 2 * np.pi, 1000)])


P = generate_P((1, 40, 1))
act_func = [np.tanh, lambda x: x]

grad = elementwise_grad(cost, 1)

N = 10000

a = 0

g1 = g2 = g3 = 0

for i in range(N):
    #print(cost(x, P, act_func, y))
    if i % (N / 100) == 0:
        print(i / (N / 100))

    gradient = grad(x, P, act_func)
    g1 = gradient[0] + a * g1
    g2 = gradient[1] + a * g2

    P[0] -= 0.001 * g1
    P[1] -= 0.001 * g2

print(cost(x, P, act_func))

plt.plot(x, g_trial(x, P, act_func))
plt.show()
