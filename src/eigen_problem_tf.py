import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tools import fig_path

tf.keras.backend.set_floatx("float64")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Set fontsizes in figures
params = {'legend.fontsize': 'large',
          'axes.labelsize': 'large',
          'axes.titlesize': 'large',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large',
          'legend.fontsize': 'large',
          'legend.handlelength': 2}
plt.rcParams.update(params)


class DNModel(tf.keras.Model):
    def __init__(self, n):
        super(DNModel, self).__init__()

        self.dense_1 = tf.keras.layers.Dense(200, activation=tf.nn.sigmoid)
        self.dense_2 = tf.keras.layers.Dense(100, activation=tf.nn.sigmoid)
        self.out = tf.keras.layers.Dense(n, name="output")

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return self.out(x)


def ray_quo(A, x):
    r = tf.matmul(tf.matmul(tf.transpose(x), A), x) / \
        tf.matmul(tf.transpose(x), x)
    return r


@tf.function
def trial_solution(model, x0, t):
    """
    Trial solution
    """
    #gtrial = (1 - t) * x0 + t * model(t)
    gtrial = tf.exp(-t) * x0 + (1 - tf.exp(-t)) * model(t)
    #gtrial = tf.exp(-t) * x0 * model(t)

    return tf.transpose(gtrial)


@tf.function
def rhs(model, A, x0, t):
    """
    Right-hand side of ODE
    """
    g = trial_solution(model, x0, t)
    F1 = tf.matmul(tf.matmul(tf.transpose(g), g) * A, g)
    F2 = tf.matmul(tf.matmul(tf.transpose(g), A), g) * g
    rhs_out = F1 - F2

    #rhs_out = tf.matmul(A, g) - tf.matmul(tf.transpose(g), g) * g

    return rhs_out


@tf.function
def loss(model, A, x0, t):
    """
    Loss/cost function
    """

    with tf.GradientTape() as tape:
        tape.watch(t)
        trial = trial_solution(model, x0, t)
        d_trial_dt = tape.gradient(trial, t)

    return tf.losses.MSE(
        tf.zeros_like(d_trial_dt), d_trial_dt - rhs(model, A, x0, t))


'''
@tf.function
def loss(model, A, x0, t):
    """
    Loss/cost function
    """
    g = trial_solution(model, x0, t)
    r = ray_quo(A, g)
    loss_ = tf.nn.l2_loss(r * g - tf.matmul(A, g))
    return loss_
'''

'''
@tf.function
def loss(model, A, x0, t):
    """
    Loss/cost function
    """
    g = trial_solution(model, x0, t)
    r = ray_quo(A, g)
    return tf.losses.MSE(tf.zeros_like(g), r * g - tf.matmul(A, g))
'''


@tf.function
def grad(model, A, x0, t):
    with tf.GradientTape() as tape:
        loss_value = loss(model, A, x0, t)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def euler(A, x0, T, N):
    """
    Euler's method
    """
    dt = T / N
    x = [x0]
    for i in range(N - 1):
        x.append(x[-1] + dt * (A @ x[-1] - (x[-1].T @ x[-1]) * x[-1]))
        # x.append(x[-1] + dt * ((x[-1].T @ x[-1]) * A @
        #                       x[-1] - (x[-1].T @ A) @ x[-1] * x[-1]))

    x = np.array(x)
    x = x / np.sqrt(np.einsum("ij,ij->i", x, x)[:, np.newaxis])
    eig = np.einsum("ij,ij->i", x @ A, x)

    return eig


if __name__ == "__main__":
    np.random.seed(42)

    n = 3
    T = 3
    N = 50

    A = np.random.normal(0, 1, (n, n))
    A = (A.T + A) / 2
    #A = -A
    x0 = np.random.rand(n)
    #x0 = x0 / np.linalg.norm(x0, ord=1)
    #x0 = np.array([1, 0, 0, 0])
    t = np.linspace(0, T, N)
    eig_euler = euler(A, x0, T, N)

    A_tf = tf.convert_to_tensor(A, dtype=tf.float64)
    x0_tf = tf.convert_to_tensor(x0, dtype=tf.float64)
    t_tf = tf.convert_to_tensor(t, dtype=tf.float64)

    # Initial model and optimizer

    model = DNModel(n)
    optimizer = tf.keras.optimizers.Nadam(0.001)

    eigvals_nn = []

    num_epochs = 40

    for t_ in t:
        t_ = tf.reshape(t_, [-1, 1])
        for epoch in range(num_epochs):

            cost, gradients = grad(model, A_tf, x0_tf, t_)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))

            print(f"Step: {optimizer.iterations.numpy()}, "
                  + f"Loss: {tf.reduce_mean(cost.numpy())}")

            #g_nn = trial_solution(model, x0_tf, t_)
            #eig_nn = ray_quo(A_tf, g_nn)
            # eigvals_nn.append(eig_nn.numpy()[0][0])

        g_nn = trial_solution(model, x0_tf, t_)
        eig_nn = ray_quo(A_tf, g_nn)
        eigvals_nn.append(eig_nn.numpy()[0][0])

    v, w = np.linalg.eig(A)
    print('A =', A)
    print('x0 =', x0)
    print('Eigvals:', v)

    '''
    iter_tot = num_epochs * N
    eig_euler2 = euler(A, x0, T, iter_tot)
    iter_plot = np.linspace(1, iter_tot, iter_tot)
    fig, ax = plt.subplots()
    ax.axhline(np.max(v), color='red', ls='--')
    ax.plot(iter_plot, eig_euler2)
    ax.plot(iter_plot, eigvals_nn)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Eigenvalue, $\\lambda$')
    lgd_numpy = "Numpy $\\lambda_{\\mathrm{max}} \\sim$ " + \
        str(round(np.max(v), 5))
    lgd_euler = "Euler $\\lambda_{\\mathrm{max}} \\sim$ " + \
        str(round(np.max(eig_euler), 5))
    lgd_nn = "FFNN $\\lambda_{\\mathrm{max}} \\sim$ " + \
        str(round(max(eigvals_nn), 5))
    plt.legend([lgd_numpy, lgd_euler, lgd_nn], loc='best')
    plt.show()
    '''

    fig, ax = plt.subplots()
    ax.axhline(np.max(v), color='red', ls='--')
    ax.plot(t, eig_euler)
    ax.plot(t, eigvals_nn)
    ax.set_xlabel('Time, $t$')
    ax.set_ylabel('Eigenvalue, $\\lambda$')
    lgd_numpy = "Numpy $\\lambda_{\\mathrm{max}} \\sim$ " + \
        str(round(np.max(v), 5))
    lgd_euler = "Euler $\\lambda_{\\mathrm{max}} \\sim$ " + \
        str(round(np.max(eig_euler), 5))
    lgd_nn = "FFNN $\\lambda_{\\mathrm{max}} \\sim$ " + \
        str(round(max(eigvals_nn), 5))
    plt.legend([lgd_numpy, lgd_euler, lgd_nn], loc='best')
    plt.show()
