import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

in_units = 784
h1_units = 300
N_class = 10
X = tf.placeholder(tf.float32, [None, in_units])
Y = tf.placeholder(tf.float32, [None, N_class])
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units], dtype=tf.float32))
W2 = tf.Variable(tf.zeros([h1_units, N_class]), dtype=tf.float32)
b2 = tf.Variable(tf.zeros([N_class], dtype=tf.float32))
keep_prob = tf.placeholder(tf.float32)

hidden1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
Y_ = tf.nn.softmax(tf.add(tf.matmul(hidden1_drop, W2), b2))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(Y_), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

correct_prediction = tf.equal( tf.argmax(Y,1), tf.argmax(Y_,1) )
accuracy = tf.reduce_mean( tf.cast(correct_prediction, tf.float32) )

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

loopN = 3000
batch_size = 128
disp_step = 10
eval_step = 100
for i in range(loopN):
    X_batch, Y_batch = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={X: X_batch, Y: Y_batch, keep_prob:0.75})
    # print(i)

    if i % disp_step == 0:
        # print("cost:",{.})
        pass

    if i % eval_step == 0:
        print(accuracy.eval({X:mnist.test.images, Y:mnist.test.labels, keep_prob:1.0}))


