import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K

tf.keras.backend.set_floatx("float64")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class DNModel(tf.keras.Model):
    def __init__(self):
        super(DNModel, self).__init__()

        self.dense_1 = tf.keras.layers.Dense(
            50, activation=tf.nn.sigmoid)
        self.out = tf.keras.layers.Dense(3, name="output")

    def call(self, inputs):
        x = self.dense_1(inputs)

        return self.out(x)


@tf.function
def g_t(model, t):
    """
    Trial solution
    """

    gtrial = tf.einsum('i...,j->ij', (1 + t), x0) + tf.einsum('i...,ij->ij', t, model(t))
    return gtrial


@tf.function
def rhs(model, t):
    """
    Right-hand side of ODE
    """

    g = g_t(model, t)
    A = A_mat()
    F1 = tf.einsum('ij,ij,kl,il->ik',g,g,A,g)
    F2 = tf.einsum('ij,jk,ik,il->il',g,A,g,g)
    return F1 - F2


@tf.function
def loss(model, t):
    """
    Loss/cost function
    """

    with tf.GradientTape() as tape:
        tape.watch(t)
        trial = g_t(model, t)
    d_trial_dt = tape.batch_jacobian(trial, t)
    d_trial_dt = d_trial_dt[:,:,0]

    return tf.losses.MSE(d_trial_dt, rhs(model, t))


# Define gradient method
@tf.function
def grad(model, t):
    with tf.GradientTape() as tape:
        loss_value = loss(model, t)

    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def ray_quo(A, x):
    x = x/np.sqrt(np.einsum("ij,ij->i", x, x)[:,np.newaxis])
    eig = np.einsum("ij,ij->i", x@A, x)
    return eig

def f(A, x):
    return (x.T@x)*A@x - (x.T@A@x)*x

def euler(A, x0, t0, t1, n=10001):
    dt = (t1 - t0)/n
    x = [x0]
    for i in range(n-1):
        x.append(x[-1] + dt*f(A, x[-1]))
    x = np.array(x)

    def euler_solution(t):
        t = t - t0
        return x[(t*(n-1)).astype(int),:]

    return euler_solution

if __name__ == "__main__":
    A = np.array([[3., 2., 4.], [2., 0., 2.], [4., 2., 3.]])

    euler_solution = euler(A, np.array([1, 0, 0]), 0, 1)
    t = np.linspace(0, 1, 100)

    fig = plt.figure()
    plt.axis([0, 1, 0, 1])
    
    x_euler = euler_solution(t)
    plt.plot(t,x_euler[:,0], linestyle='--', color='b')
    plt.plot(t,x_euler[:,1], linestyle='--', color='g')
    plt.plot(t,x_euler[:,2], linestyle='--', color='r')
 
    def A_mat():
        A = tf.constant([[3., 2., 4.], [2., 0., 2.], [
                        4., 2., 3.]], dtype=tf.float64)
        return A

    Nt = 21   # number of time points
    # x0 = tf.constant([0.2, 0.3, 0.7], dtype=tf.float64)  # initial position
    x0 = tf.constant([1, 0, 0], dtype=tf.float64)  # initial position

    start = tf.constant(0, dtype=tf.float64)
    stop = tf.constant(1, dtype=tf.float64)
    t = tf.linspace(start, stop, Nt)
    t = tf.reshape(t, [-1, 1])

    # Initial model and optimizer
    model = DNModel()
    optimizer = tf.keras.optimizers.Adam(0.01)
    num_epochs = 100000
    # g = g_t(model,t)
    # nn_graph_0 = plt.plot(t, g[:,0], color='b')
    # nn_graph_1 = plt.plot(t, g[:,1], color='g')
    # nn_graph_2 = plt.plot(t, g[:,2], color='r')
    for epoch in range(num_epochs):
            cost, gradients = grad(model, t)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))

            print(f"Step: {optimizer.iterations.numpy()}, "
                + f"Loss: {tf.math.reduce_mean(cost.numpy())}")
            # nn_graph_0.pop(0).remove()
            # nn_graph_1.pop(0).remove()
            # nn_graph_2.pop(0).remove()
            # g = g_t(model,t)
            # nn_graph_0 = plt.plot(t, g[:,0], color='b')
            # nn_graph_1 = plt.plot(t, g[:,1], color='g')
            # nn_graph_2 = plt.plot(t, g[:,2], color='r')
            # print(print(loss2(model, t)))
            # plt.pause(0.0001)
            
    g = g_t(model,t)
    nn_graph_0 = plt.plot(t, g[:,0], color='b')
    nn_graph_1 = plt.plot(t, g[:,1], color='g')
    nn_graph_2 = plt.plot(t, g[:,2], color='r')
    plt.show()

    # Anumpy = np.array([[3., 2., 4.], [2., 0., 2.], [4., 2., 3.]])
    # w, v = np.linalg.eig(Anumpy)
    # print('Numpy eigval:\n', w)
    # print('Numpy eigvec:\n', v)
