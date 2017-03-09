import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def init_weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def init_bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pooling(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

N_input = 784
c1_size = 32
c2_size = 64
f1_unit = 1024
f2_unit = 10
X = tf.placeholder(tf.float32, [None, N_input])
Y = tf.placeholder(tf.float32, [None, f2_unit])
X_in = tf.reshape(X, [-1, 28, 28, 1])

# define weights and init_bias
w1_conv = init_weight([5, 5, 1, c1_size])
b1_conv = init_bias([c1_size])
w2_conv = init_weight([5, 5, c1_size, c2_size])
b2_conv = init_bias([c2_size])
w1_fc = init_weight([7*7*64, f1_unit])
b1_fc = init_bias([f1_unit])
w2_fc = init_weight([f1_unit, f2_unit])
b2_fc = init_bias([f2_unit])

c1_conv = tf.nn.relu(conv2d(X_in, w1_conv) + b1_conv)
s2_pool = max_pooling(c1_conv)
c3_conv = tf.nn.relu(conv2d(s2_pool, w2_conv) + b2_conv)
s4_pool = max_pooling(c3_conv)
c5_in = tf.reshape(s4_pool,[-1, 7*7*64])
c5_fc = tf.nn.relu(tf.matmul(c5_in, w1_fc) + b1_fc)
keep_prob = tf.placeholder(tf.float32)
c5_dropout = tf.nn.dropout(c5_fc, keep_prob)
y_ = tf.nn.softmax(tf.matmul(c5_dropout, w2_fc) + b2_fc)

# cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_)))

# optimizer
step_train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

correct_prediction = tf.equal(tf.arg_max(Y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# start training...
for i in range(3000):
    X_batch, Y_batch = mnist.train.next_batch(100)
    step_train.run({X:X_batch, Y:Y_batch, keep_prob:0.5})

    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict = {X:X_batch, Y:Y_batch, keep_prob:0.5})
        print("step %d, training accuracy %g"%(i, train_accuracy))

# final accuracy
print("test accuracy %g"%accuracy.eval(feed_dict={X:mnist.test.images, Y:mnist.test.labels, keep_prob:1.0}))

