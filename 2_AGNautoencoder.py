import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import sklearn.preprocessing as prep

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

# define class
class AGNautoencoder(object):
    def __init__(self, inputN, hideN, transfer_fun=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        self.inputN = inputN
        self.hideN = hideN
        self.transfer = transfer_fun
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        self.weights = self._initializer_weights()
        self.x = tf.placeholder(tf.float32, [None, self.inputN])
        self.hide = self.transfer(tf.add(tf.matmul(
            self.x + self.scale * tf.random_normal((self.inputN,)),
            self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hide, self.weights['w2']), self.weights['b2'])
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)
        init = tf.global_variables_initializer() #attention! don't foget it
        self.sess = tf.Session()
        self.sess.run(init)

    def _initializer_weights(self):
        weights = dict()
        weights['w1'] = tf.Variable(xavier_init(self.inputN, self.hideN))
        weights['b1'] = tf.Variable(tf.zeros([self.hideN], dtype=tf.float32))
        weights['w2'] = tf.Variable(tf.zeros([self.hideN, self.inputN], dtype=tf.float32))
        weights['b2'] = tf.Variable(tf.zeros([self.inputN], dtype=tf.float32))
        return weights

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer),
                                  feed_dict={self.x: X, self.scale: self.training_scale})
        return cost

    def cal_cost(self, X):
        cost = self.sess.run(self.cost, 
                             feed_dict={self.x: X, self.scale: self.training_scale})
        return cost

    def gethide(self, X):
        hide = self.sess.run(self.hide, 
                             feed_dict={self.x: X, self.scale: self.training_scale})
        return hide

    def generate(self, hide=None):
        if hide is None:
            hide = np.random.normal(size=self.weights['b1'])
        reconstruction = self.sess.run(self.reconstruction, feed_dict={self.hide: hide})
        return reconstruction

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction,
                             feed_dict={self.x: X, self.scale: self.training_scale})

def data_standard(X_train, X_test):
    processor = prep.StandardScaler().fit(X_train)
    X_train = processor.transform(X_train)
    X_test = processor.transform(X_test)
    return X_train, X_test

def getbatches(data, batch_size):
    k = np.random.randint(0, len(data) - batch_size)
    return data[k:(k + batch_size)]

epoch_size = 20
batch_size = 128
display_intval = 1
autoencoder = AGNautoencoder(inputN=784,
                             hideN=200,
                             transfer_fun=tf.nn.softplus,
                             optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                             scale=0.01)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_data, test_data = data_standard(mnist.train.images, mnist.test.images)
samples = int(mnist.train.num_examples)


for epoch in range(epoch_size):
    avg_cost = 0.
    batch_num = int(samples / batch_size)
    print(samples)
    print(batch_num)
    for loop in range(batch_num):
        X_batch = getbatches(train_data, batch_size)
        cost = autoencoder.partial_fit(X_batch)
        avg_cost += cost/samples*batch_size
        # print(loop, ":", avg_cost)

    if epoch % display_intval == 0:
        print("epoch:", '%04d' % (epoch+1), "cost:", "{:.9f}".format(avg_cost))
