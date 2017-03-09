import cifar10, cifar10_input
import tensorflow as tf
import numpy as np
import time
import math

batch_size = 128
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'

cifar10.maybe_download_and_extract()
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        tf.add_to_collection('losses',weight_loss)
    return var

images_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
labels_holder = tf.placeholder(tf.int32, [batch_size])

# fisrt layer: convolution
W1 = variable_with_weight_loss([5, 5, 3, 64], 5e-2, 0.0)
b1 = tf.Variable(tf.zeros([64]))
c1_conv = tf.nn.relu(tf.nn.conv2d(images_holder, W1, strides=[1, 1, 1, 1], padding='SAME') + b1)
c1_pool = tf.nn.max_pool(c1_conv, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
c1_LRN = tf.nn.lrn(c1_pool, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

# second layer: convolution
W2 = variable_with_weight_loss([5, 5, 64, 64], 5e-2, 0.0)
b2 = tf.Variable(tf.constant(0.1, shape=[64]))
c2_conv = tf.nn.relu(tf.nn.conv2d(c1_LRN, W2, strides=[1, 1, 1, 1], padding='SAME') + b2)
c2_LRN = tf.nn.lrn(c2_conv, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
c2_pool = tf.nn.max_pool(c2_LRN, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# third layer: full connection
f3_in = tf.reshape(c2_pool, [batch_size, -1])
dim = f3_in.get_shape()[1].value
W3 = variable_with_weight_loss([dim, 384], 4e-2, 0.004)
b3 = tf.Variable(tf.constant(0.1, shape=[384]))
f3_fc = tf.nn.relu(tf.matmul(f3_in, W3) + b3)

# fourth layer: full connection
W4 = variable_with_weight_loss([384, 192], 0.04, 0.004)
b4 = tf.Variable(tf.constant(0.1, shape=[192]))
f4_fc = tf.nn.relu(tf.matmul(f3_fc, W4) + b4)

# last layer: full connection
W5 = variable_with_weight_loss([192, 10], 1 / 192.0, 0.0)
b5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.matmul(f4_fc, W5) + b5

# loss and optimizer
def losses(logit, labels):
    labels = tf.cast(labels, tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logit, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

total_loss = losses(logits, labels_holder)
train_step = tf.train.AdamOptimizer(1e-3).minimize(total_loss)

top_k_op = tf.nn.in_top_k(logits, labels_holder, 1)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

tf.train.start_queue_runners()

# start training...
max_steps = 3000
disp_step = 10
for step in range(max_steps):
    image_batch,label_batch = sess.run((images_train, labels_train))
    start_time = time.time()
    _, loss_value = sess.run([train_step, total_loss], feed_dict={images_holder: image_batch, labels_holder: label_batch})
    duration = time.time() - start_time
    if step % disp_step == 0:
        example_per_second = batch_size / duration
        second_per_batch = float(duration)
        format_str = "step %d, loss=%.2f (%.1f example/sec; %.3f sec/batch)"
        print(format_str % (step, loss_value, example_per_second, second_per_batch))

#

num_examples = 10000
num_iter = int(math.ceil(num_examples / batch_size))
total_sample_count = num_iter * batch_size
true_count = 0
for step in range(num_iter):
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run(top_k_op, feed_dict={images_holder: image_batch, labels_holder: label_batch})
    true_count += np.sum(predictions)
    print("step: %.3f" % true_count)

precision = 1.0 * true_count / total_sample_count
print('precision @ 1 = {:.3f}'.format(precision))




