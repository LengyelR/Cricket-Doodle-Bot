import tensorflow as tf


class ConvNet:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 196], name="input_x")
        self.y_ = tf.placeholder(tf.float32, [None, 10], name="output_y")

        self.y_conv, self.keep_prob = self.network()
        self.opt = self.optimiser()
        self.accuracy = self.loss()

    def network(self):
        def conv2d(x_inner, w, name):
            return tf.nn.conv2d(x_inner, w, strides=[1, 1, 1, 1], padding='SAME', name=name)

        def max_pool_2x2(x_inner, name):
            return tf.nn.max_pool(x_inner, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME', name=name)

        def weight_variable(shape, name):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name=name)

        def bias_variable(shape, name):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=name)

        with tf.name_scope('reshape'):
            x_image = tf.reshape(self.x, [-1, 14, 14, 1])

        with tf.name_scope('conv1'):
            W_conv1 = weight_variable([3, 3, 1, 16], 'w1')
            b_conv1 = bias_variable([16], 'b1')
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 'h1_inner') + b_conv1, 'h1')

        with tf.name_scope('pool1'):
            h_pool1 = max_pool_2x2(h_conv1, 'pool1')

        with tf.name_scope('conv2'):
            W_conv2 = weight_variable([5, 5, 16, 32], 'w2')
            b_conv2 = bias_variable([32], 'b2')
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 'h2_inner') + b_conv2, 'h2')

        with tf.name_scope('pool2'):
            h_pool2 = max_pool_2x2(h_conv2, 'pool2')

        with tf.name_scope('fc1'):
            W_fc1 = weight_variable([4 * 4 * 32, 512], 'w3_full')
            b_fc1 = bias_variable([512], 'b3_full')

            h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 4 * 32], name='h3_reshape')
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name='h3_full')

        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        with tf.name_scope('fc2'):
            W_fc2 = weight_variable([512, 10], 'w4_out')
            b_fc2 = bias_variable([10], 'b4_out')

            y_conv = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2, name='y_conv_model')
        return y_conv, keep_prob

    def optimiser(self):
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)

        with tf.name_scope('adam_optimizer'):
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        return train_step

    def loss(self):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        return accuracy
