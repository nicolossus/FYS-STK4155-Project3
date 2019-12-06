#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.keras.backend.set_floatx("float64")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class DNModel(tf.keras.Model):
    def __init__(self, n):
        super(DNModel, self).__init__()

        self.dense_1 = tf.keras.layers.Dense(100, activation=tf.nn.sigmoid)
        self.dense_2 = tf.keras.layers.Dense(50, activation=tf.nn.sigmoid)
        self.dense_3 = tf.keras.layers.Dense(25, activation=tf.nn.sigmoid)
        self.out = tf.keras.layers.Dense(n, name="output")

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return self.out(x)


@tf.function
def trial_solution(model, x0, t):
    """Trial solution"""
    gtrial = tf.einsum('i...,j->ij', tf.exp(-t), x0) + \
        tf.einsum('i...,ij->ij', (1 - tf.exp(-t)), model(t))
    return gtrial


@tf.function
def rhs(model, A, x0, t):
    """ODE right-hand side"""
    g = trial_solution(model, x0, t)
    # ode_rhs = tf.einsum('ij,ij,kl,il->ik', g, g, A, g) - \
    #    tf.einsum('ij,jk,ik,il->il', g, A, g, g)

    ode_rhs = tf.einsum('jk,ik->ij', A, g) - tf.einsum('ij,ij,ik->ik', g, g, g)
    return ode_rhs


@tf.function
def loss(model, A, x0, t):
    """Loss function"""
    with tf.GradientTape() as tape:
        tape.watch(t)
        trial = trial_solution(model, x0, t)
    d_trial_dt = tape.batch_jacobian(trial, t)
    d_trial_dt = d_trial_dt[:, :, 0]
    return tf.losses.MSE(d_trial_dt, rhs(model, A, x0, t))


@tf.function
def grad(model, A, x0, t):
    """Gradient method"""
    with tf.GradientTape() as tape:
        loss_value = loss(model, A, x0, t)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def ray_quo(A, x):
    """Rayleigh quotient"""
    x = x / tf.sqrt(tf.einsum("ij,ij->i", x, x)[:, tf.newaxis])
    return tf.einsum("ij,ij->i", tf.matmul(x, A), x)

# Define Euler's method


def euler_eig(A, x0, T, N):
    """Euler's method"""
    dt = T / N
    x = [x0]
    for i in range(N - 1):
        x.append(x[-1] + dt * (A @ x[-1] - (x[-1].T @ x[-1]) * x[-1]))
        # x.append(x[-1] + dt * ((x[-1].T @ x[-1]) * A @
        #                       x[-1] - (x[-1].T @ A) @ x[-1] * x[-1]))

    x = np.array(x)
    x = x / np.sqrt(np.einsum("ij,ij->i", x, x)[:, np.newaxis])
    eig = np.einsum("ij,ij->i", x @ A, x)

    return x, eig


if __name__ == "__main__":

    tf.random.set_seed(32)
    np.random.seed(32)

    # Define problem
    n = 6    # Dimension
    T = 5  # Final time
    N = 1001  # number of time points

    # Benchmark problem
    A = np.random.normal(0, 1, (n, n))
    A = (A.T + A) * 0.5
    x0 = np.random.rand(n)
    x0 = x0 / np.linalg.norm(x0, ord=1)
    t = np.linspace(0, T, N)

    # Problem formulation for tensorflow
    Nt = 6
    A_tf = tf.convert_to_tensor(A, dtype=tf.float64)
    x0_tf = tf.convert_to_tensor(x0, dtype=tf.float64)
    start = tf.constant(0, dtype=tf.float64)
    stop = tf.constant(T, dtype=tf.float64)
    t_tf = tf.linspace(start, stop, Nt)
    t_tf = tf.reshape(t_tf, [-1, 1])

    # Initial model and optimizer
    model = DNModel(n)
    optimizer = tf.keras.optimizers.Adam(0.01)
    num_epochs = 2000

    for epoch in range(num_epochs):
        cost, gradients = grad(model, A, x0_tf, t_tf)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        step = optimizer.iterations.numpy()
        if step == 1:
            print(f"Step: {step}, " +
                  f"Loss: {tf.math.reduce_mean(cost.numpy())}")
        if step % 100 == 0:
            print(f"Step: {step}, " +
                  f"Loss: {tf.math.reduce_mean(cost.numpy())}")

    # Call models
    x_euler, eig_euler = euler_eig(A, x0, T, N)

    g = trial_solution(model, x0_tf, t_tf)
    eig_nn = ray_quo(A_tf, g)

    # Print results
    v, w = np.linalg.eig(A)
    print()
    print('A =', A)
    print('x0 =', x0)
    print('Eigvals Numpy:', v)
    print('Max Eigval Numpy', np.max(v))
    print('Final Eigval Euler', eig_euler[-1])
    print('Final Eigval FFNN', eig_nn.numpy()[-1])

    # Plot eigenvalues
    fig, ax = plt.subplots()
    ax.axhline(np.max(v), color='red', ls='--')
    ax.plot(t, eig_euler)
    ax.plot(t_tf, eig_nn)
    ax.set_xlabel('Time, $t$')
    ax.set_ylabel('Eigenvalue, $\\lambda$')
    lgd_numpy = "Numpy $\\lambda_{\\mathrm{max}} \\sim$ " + \
        str(round(np.max(v), 5))
    lgd_euler = "Euler $\\lambda_{\\mathrm{final}} \\sim$ " + \
        str(round(eig_euler[-1], 5))
    lgd_nn = "FFNN $\\lambda_{\\mathrm{final}} \\sim$ " + \
        str(round(eig_nn.numpy()[-1], 5))
    plt.legend([lgd_numpy, lgd_euler, lgd_nn], loc='best')
    plt.show()
