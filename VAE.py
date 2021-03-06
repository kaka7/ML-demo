# encoding=utf-8
import os
import math
import numpy as np
import tensorflow as tf
import  matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
stdd = 0.1

def get_mnist_data():
    mnist = input_data.read_data_sets('/home/naruto/PycharmProjects/data/mnist',one_hot=True)
    return mnist

def show_img(test_data,test_y, mean_img,nimgs=10):
    fig, axs = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(nimgs):
        axs[0][i].imshow(
            np.reshape(test_data[i, :], (28, 28)))
        axs[1][i].imshow(
            np.reshape([test_y[i, :] + mean_img], (28, 28)))

    fig.show()
    plt.waitforbuttonpress()

class VAEModel:

    def __init__(self):
        pass

    def initWeight(self, shape):
        high = np.sqrt(6.0 / (shape[0] + shape[1]))
        low = -1 * high
        return tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32)

    def encoder(self, input, numOuts=[500, 200, 20]):
        current_x = input
        #        current_x = get_noise_image(input, 0.5)
        for l, n_out in enumerate(numOuts[:-1]):
            n_in = int(current_x.get_shape()[1])#784
            shape = [n_in, n_out]
            w = tf.Variable(self.initWeight(shape))
            b = tf.Variable(tf.zeros([n_out]))
            hidden = tf.nn.relu(tf.matmul(current_x, w) + b)
            current_x = hidden#batch——size X２０

        shape = [n_out, numOuts[-1]]
        w_var = tf.Variable(self.initWeight(shape))
        b_var = tf.Variable(tf.zeros([numOuts[-1]]))
        w_mean = tf.Variable(self.initWeight(shape))
        b_mean = tf.Variable(tf.zeros([numOuts[-1]]))

        z_var = tf.matmul(hidden, w_var) + b_var
        z_mean = tf.matmul(hidden, w_mean) + b_mean
        return (z_mean, z_var)

    # 采样一个标准高斯分布，并通过encoder学习到的参数，生成z
    def sample_z(self, z_mean, z_var, std=1.0):

        epsilon = tf.random_normal(z_var.get_shape(), mean=0, stddev=stdd)
        z = tf.multiply(tf.exp(0.5 * z_var), epsilon) + z_mean
        return z

    def decoder(self, input, numOuts=[200, 500, 784]):
        current_h = input
        for l, n_out in enumerate(numOuts):
            n_in = int(current_h.get_shape()[1])
            shape = [n_in, n_out]
            w = tf.Variable(self.initWeight(shape))
            b = tf.Variable(tf.zeros([n_out]))
            if (l == len(numOuts) - 1):
                hidden = tf.nn.sigmoid(tf.matmul(current_h, w) + b)
            else:
                hidden = tf.nn.relu(tf.matmul(current_h, w) + b)
            current_h = hidden
        return hidden

    def loss(self, x, y, z_mean, z_var):
        entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=x)
        #        entropy = tf.square(x-y)
        loss = tf.reduce_sum(entropy, 1)
        kl_loss = -0.5 * tf.reduce_sum(1 + z_var - tf.square(z_mean) - tf.exp(z_var), 1)
        all_loss = tf.reduce_mean(loss + kl_loss)
        return all_loss

    def train(self, lr, loss):
        optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
        return optimizer


def train_VAE():
    mnist = get_mnist_data()
    mean_img = np.mean(mnist.train.images, axis=0)#1X７８４
    batch_size = 100
    epochs = 10
    lr = 0.003
    x = tf.placeholder(tf.float32, shape=[batch_size, 784])
    print(x.get_shape())
    vae = VAEModel()
    mean, var = vae.encoder(x)#１００X２０
    z = vae.sample_z(mean, var)
    y = vae.decoder(z)
    loss = vae.loss(x, y, mean, var)
    opt = vae.train(lr, loss)
    #    opt = tf.train.AdamOptimizer(lr).minimize(loss)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    for epoch in range(epochs):
        for i in range(mnist.train.num_examples // batch_size):
            batch = mnist.train.next_batch(batch_size)[0]
            train_x = np.array([(img - mean_img) for img in batch])
            feed_dict = {x: train_x}
            y_loss, _ = sess.run([loss, opt], feed_dict=feed_dict)
        print("epoch: %d ,loss: %f" % (epoch, y_loss))
    test_data = mnist.test.next_batch(batch_size)[0]
    test_x = np.array([(img - mean_img) for img in test_data])
    test_y = sess.run(y, feed_dict={x: test_x})
    show_img(test_data, test_y, mean_img)


if __name__ == '__main__':
    train_VAE()