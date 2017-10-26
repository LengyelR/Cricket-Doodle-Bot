import tensorflow as tf


class LoadModel:
    def __init__(self, loc, output, name, input_layer='input_x', dropout=True):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.input = input_layer + ':0'
        self.dropout = dropout

        with self.graph.as_default():
            saver = tf.train.import_meta_graph(loc + '/' + name + '.ckpt.meta', clear_devices=True)
            saver.restore(self.sess, tf.train.latest_checkpoint(loc))

            self.output = self.graph.get_tensor_by_name(output + ':0')

    def run(self, data):
        if self.dropout:
            return self.sess.run(self.output, feed_dict={self.input: data, "dropout/keep_prob:0": 1.0})
        else:
            return self.sess.run(self.output, feed_dict={self.input: data})


class Policy:
    def __init__(self):
        self.input = tf.placeholder(tf.float32, shape=[None, 23 * 25 * 1], name='input_x')
        self.actions = tf.placeholder(tf.int32)
        self.rewards = tf.placeholder(tf.float32)

        self.y_ = self.build_model()
        self.opt, self.loss = self.optimiser()
        self.act = self.action()
        self.debug_all_var = tf.trainable_variables()

    def build_model(self):
        # TODO: baseclass
        def weight_variable(shape, name):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name=name)

        def bias_variable(shape, name):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=name)

        with tf.name_scope('fc1'):
            W1 = weight_variable([23 * 25 * 1, 300], 'w1')
            b1 = bias_variable([300], 'b1')
            h_fc1 = tf.nn.relu(tf.matmul(self.input, W1) + b1, name='h1')

        with tf.name_scope('softmax'):
            W_fc2 = weight_variable([300, 2], 'fc2')
            b2 = bias_variable([2], 'b2')
            return tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b2)

    def action(self):
        with tf.name_scope('output'):
            logits = tf.log(self.y_ / (1 - self.y_))
            return tf.squeeze(tf.multinomial(logits, 1), name='action')

    def optimiser(self):
        with tf.name_scope('loss'):
            indices = tf.range(0, tf.shape(self.y_)[0]) * tf.shape(self.y_)[1] + self.actions
            chosen_action_probs = tf.gather(tf.reshape(self.y_, [-1]), indices)
            loss = -tf.reduce_mean(tf.log(chosen_action_probs) * self.rewards)
        with tf.name_scope('opt'):
            optimiser = tf.train.AdamOptimizer(learning_rate=1e-4)
            return optimiser.minimize(loss), loss
