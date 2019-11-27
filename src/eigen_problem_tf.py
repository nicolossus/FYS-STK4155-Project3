import os
import tensorflow as tf
import numpy as np

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

    #gtrial = (1 - t) * x0 + t * model(t)
    gtrial = tf.exp(-t) * x0 + (1 - tf.exp(-t)) * model(t)
    gtrial = tf.transpose(gtrial)
    return gtrial


@tf.function
def rhs(model, t):
    """
    Right-hand side of ODE
    """

    g = g_t(model, t)
    A = A_mat()
    F1 = tf.matmul(tf.matmul(tf.matmul(tf.transpose(g), g), A), g)
    F2 = tf.matmul(tf.matmul(tf.matmul(tf.transpose(g), A), g), g)
    return F1 - F2


@tf.function
def loss(model, t):
    """
    Loss/cost function
    """

    with tf.GradientTape() as tape:
        tape.watch(t)
        trial = g_t(model, t)
        d_trial_dt = tape.gradient(trial, t)

    return tf.losses.MSE(
        tf.zeros_like(d_trial_dt), d_trial_dt - rhs(model, trial))


# Define gradient method
@tf.function
def grad(model, t):
    with tf.GradientTape() as tape:
        loss_value = loss(model, t)

    return loss_value, tape.gradient(loss_value, model.trainable_variables)


if __name__ == "__main__":

    def A_mat():
        A = tf.constant([[3., 2., 4.], [2., 0., 2.], [
                        4., 2., 3.]], dtype=tf.float64)
        return A

    Nt = 21   # number of time points
    # x0 = tf.constant([0.2, 0.3, 0.7], dtype=tf.float64)  # initial position
    x0 = tf.constant([0.6, 0.3, 0.6], dtype=tf.float64)  # initial position

    start = tf.constant(0, dtype=tf.float64)
    stop = tf.constant(10, dtype=tf.float64)
    t = tf.linspace(start, stop, Nt)

    # Initial model and optimizer
    model = DNModel()
    optimizer = tf.keras.optimizers.Adam(0.01)

    num_epochs = 100
    for t_ in t:
        t_ = tf.reshape(t_, [-1, 1])
        for epoch in range(num_epochs):
            cost, gradients = grad(model, t_)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))

            # print(f"Step: {optimizer.iterations.numpy()}, "
            #    + f"Loss: {tf.reduce_mean(cost.numpy())}")

        g_nn = g_t(model, t_)
        eigval = tf.matmul(tf.matmul(tf.transpose(g_nn), A_mat()),
                           g_nn) / tf.matmul(tf.transpose(g_nn), g_nn)

        print('Time:', t_.numpy()[0][0])
        print('Eigvec:', '(' + str(g_nn.numpy()[0][0]) + ', ' + str(
            g_nn.numpy()[1][0]) + ', ' + str(g_nn.numpy()[2][0]) + ')')
        print('Eigval:', eigval.numpy()[0][0])
        print()

    Anumpy = np.array([[3., 2., 4.], [2., 0., 2.], [4., 2., 3.]])
    w, v = np.linalg.eig(Anumpy)
    print('Numpy eigval:\n', w)
    print('Numpy eigvec:\n', v)
