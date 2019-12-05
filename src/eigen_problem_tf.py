#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K

tf.keras.backend.set_floatx("float32")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class DNModel(tf.keras.Model):
    def __init__(self, n):
        super(DNModel, self).__init__()

        self.dense_1 = tf.keras.layers.Dense(400, activation=tf.nn.sigmoid)
        self.dense_2 = tf.keras.layers.Dense(300, activation=tf.nn.relu)
        self.dense_3 = tf.keras.layers.Dense(200, activation=tf.nn.relu)
        self.dense_4 = tf.keras.layers.Dense(100, activation=tf.nn.relu)
        self.dense_5 = tf.keras.layers.Dense(50, activation=tf.nn.sigmoid)
        self.out = tf.keras.layers.Dense(n, name="output")

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        x = self.dense_5(x)

        return self.out(x)


@tf.function
def trial_solution(model, x0, t):
    """
    Trial solution
    """

    gtrial = tf.einsum('i...,j->ij', (1 + t), x0) + \
        tf.einsum('i...,ij->ij', t, model(t))

    #gtrial = tf.cast(gtrial, tf.float32)
    return gtrial


@tf.function
def rhs(model, A, x0, t):
    """
    Right-hand side of ODE
    """
    A = tf.cast(A, tf.float32)
    g = trial_solution(model, x0, t)
    # Version 1
    #F1 = tf.einsum('ij,ij,kl,il->ik', g, g, A, g)
    #F2 = tf.einsum('ij,jk,ik,il->il', g, A, g, g)

    # Version 2
    F1 = tf.einsum('jk,ik->ij', A, g)
    F2 = tf.einsum('ij,ij,ik->ik', g, g, g)

    rhs_out = F1 - F2

    return rhs_out


@tf.function
def loss(model, A, x0, t):
    """
    Loss/cost function
    """

    with tf.GradientTape() as tape:
        tape.watch(t)
        trial = trial_solution(model, x0, t)
    d_trial_dt = tape.batch_jacobian(trial, t)
    d_trial_dt = d_trial_dt[:, :, 0]

    #loss_out = tf.losses.MSE(d_trial_dt, rhs(model, A, x0, t))
    loss_out = tf.losses.MSE(
        tf.zeros_like(d_trial_dt), d_trial_dt - rhs(model, A, x0, t))

    return loss_out


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

# Define gradient method
@tf.function
def grad(model, A, x0, t):
    with tf.GradientTape() as tape:
        loss_value = loss(model, A, x0, t)

    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def ray_quo_tf(A, x):
    x = x / tf.sqrt(tf.einsum("ij,ij->i", x, x)[:, tf.newaxis])
    eig = tf.einsum("ij,ij->i", tf.matmul(x, A), x)
    return eig


def ray_quo_np(A, x):
    x = x / np.sqrt(np.einsum("ij,ij->i", x, x)[:, np.newaxis])
    eig = np.einsum("ij,ij->i", x@A, x)
    return eig


def f(A, x):
    return (x.T@x) * A@x - (x.T@A@x) * x


def euler(A, x0, t0, t1, n=10001):
    dt = (t1 - t0) / n
    x = [x0]
    for i in range(n - 1):
        x.append(x[-1] + dt * f(A, x[-1]))
    x = np.array(x)

    def euler_solution(t):
        t = t - t0
        return x[(t * (n - 1)).astype(int), :]

    return euler_solution


def euler_eig(A, x0, T, N):
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

    n = 3    # Dimension
    T0 = 0   # Start time
    T = 20    # Final time
    N = 151  # number of time points

    # Problem formulation for numpy (Euler and eig solver)

    # Benchmark problem
    n = 3
    T = 1
    A = np.array([[3., 2., 4.], [2., 0., 2.], [4., 2., 3.]])
    #A = -A
    x0 = np.array([1, 0, 0])
    t = np.linspace(T0, T, N)

    '''
    # Random problem

    A = np.random.normal(0, 1, (n, n))
    A = (A.T + A) * 0.5
    #A = -A
    x0 = np.random.rand(n)
    x0 = x0 / np.linalg.norm(x0, ord=1)
    t = np.linspace(T0, T, N)
    '''

    # Problem formulation for tensorflow
    A_tf = tf.convert_to_tensor(A, dtype=tf.float32)
    x0_tf = tf.convert_to_tensor(x0, dtype=tf.float32)
    t_tf = tf.convert_to_tensor(t, dtype=tf.float32)
    t_tf = tf.reshape(t_tf, [-1, 1])

    # Initial model and optimizer
    model = DNModel(n)
    optimizer = tf.keras.optimizers.Adam(0.01)
    num_epochs = 1500

    for epoch in range(num_epochs):
        cost, gradients = grad(model, A, x0_tf, t_tf)
        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))

        step = optimizer.iterations.numpy()
        if step == 1:
            print(f"Step: {step}, "
                  + f"Loss: {tf.math.reduce_mean(cost.numpy())}")
        if step % 100 == 0:
            print(f"Step: {step}, "
                  + f"Loss: {tf.math.reduce_mean(cost.numpy())}")

    # Call final models

    # Just for benchmark problem
    euler_solution = euler(A, x0, T0, T)
    x_euler = euler_solution(t)
    #eig_euler = ray_quo_np(A, x_euler)

    # For all problems
    eig_euler = euler_eig(A, x0, T, N)

    g = trial_solution(model, x0_tf, t_tf)
    eig_nn = ray_quo_tf(A_tf, g)
    eigvals_nn = eig_nn.numpy()

    # Print results
    v, w = np.linalg.eig(A)
    print()
    print('A =', A)
    print('x0 =', x0)
    print('Eigvals Numpy:', v)
    print('Max Eigval Numpy', np.max(v))
    print('Final Eigval Euler', eig_euler[-1])
    print('Final Eigval FFNN', eigvals_nn[-1])

    # Plot results

    # Just for benchmark problem
    fig0, ax0 = plt.subplots()
    ax0.plot(t, x_euler[:, 0], ls='--', color='b')
    ax0.plot(t, x_euler[:, 1], ls='--', color='g')
    ax0.plot(t, x_euler[:, 2], ls='--', color='r')
    ax0.plot(t_tf, g[:, 0], color='b')
    ax0.plot(t_tf, g[:, 1], color='g')
    ax0.plot(t_tf, g[:, 2], color='r')
    ax0.set_ylabel('Eigenvector Element Value')
    ax0.set_xlabel('Time')

    # For all problems
    fig, ax = plt.subplots()
    ax.axhline(np.max(v), color='red', ls='--')
    ax.plot(t, eig_euler)
    ax.plot(t, eigvals_nn)
    ax.set_xlabel('Time, $t$')
    ax.set_ylabel('Eigenvalue, $\\lambda$')
    lgd_numpy = "Numpy $\\lambda_{\\mathrm{max}} \\sim$ " + \
        str(round(np.max(v), 5))
    lgd_euler = "Euler $\\lambda_{\\mathrm{final}} \\sim$ " + \
        str(round(eig_euler[-1], 5))
    lgd_nn = "FFNN $\\lambda_{\\mathrm{final}} \\sim$ " + \
        str(round(eigvals_nn[-1], 5))
    plt.legend([lgd_numpy, lgd_euler, lgd_nn], loc='best')
    plt.show()
