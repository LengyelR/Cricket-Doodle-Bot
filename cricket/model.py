import tensorflow as tf
import threading
from os import path


class UpdateThread(threading.Thread):
    def __init__(self, sess, agent, input, actions, rewards, idx, logfolder):
        threading.Thread.__init__(self)
        self.sess = sess
        self.agent = agent
        self.input = input
        self.actions = actions
        self.rewards = rewards
        self.idx = idx
        self.logfolder = logfolder

    def run(self):
        name = 'thread:' + str(self.idx)
        print('---------UPDATE---------')
        print(name, "running")
        if threading.active_count() > 2:
            raise Exception('previous thread is still running!')

        feed_dict = {self.agent.input: self.input,
                     self.agent.actions: self.actions,
                     self.agent.rewards: self.rewards,
                     self.agent.keep_prob: 0.8}
        self.sess.run(self.agent.opt, feed_dict=feed_dict)
        loss = self.sess.run(self.agent.loss, feed_dict=feed_dict)
        print('loss:', loss)
        with open(path.join(self.logfolder, 'log.txt'), 'a') as logf:
            logf.writelines(str(loss) + '\n')
        print(name, 'finished')


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


class ConvolutionalNetwork:
    def __init__(self, width, height, channels=1,
                 conv1_map=5, conv1_features=32,
                 conv2_map=8, conv2_features=64,
                 fc_out=512, output_features=2):
        self.input = tf.placeholder(tf.float32, [None, width * height], name="input_x")
        self.width = width
        self.height = height
        self.channels = channels
        self.conv1_features = conv1_features
        self.conv1_map = conv1_map
        self.conv2_features = conv2_features
        self.conv2_map = conv2_map
        self.fc_out = fc_out
        self.output_features = output_features

        self.y_conv, self.keep_prob = self.network()

    def network(self):
        def conv2d(x_inner, w, name):
            return tf.nn.conv2d(x_inner, w, strides=[1, 1, 1, 1], padding='SAME', name=name)

        def max_pool_2x2(x_inner, name):
            return tf.nn.max_pool(x_inner, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME', name=name)

        def variable(shape, name):
            return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

        with tf.name_scope('reshape'):
            x_reshaped = tf.reshape(self.input, [-1, self.width, self.height, self.channels])

        with tf.name_scope('conv1'):
            W_conv1 = variable([self.conv1_map, self.conv1_map, self.channels, self.conv1_features], 'w1')
            b_conv1 = variable([self.conv1_features], 'b1')
            h_conv1 = tf.nn.relu(conv2d(x_reshaped, W_conv1, 'conv1') + b_conv1)

        with tf.name_scope('pool1'):
            h_pool1 = max_pool_2x2(h_conv1, 'pool1')

        with tf.name_scope('conv2'):
            W_conv2 = variable([self.conv2_map, self.conv2_map, self.conv1_features, self.conv2_features], 'w2')
            b_conv2 = variable([self.conv2_features], 'b2')
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 'conv2') + b_conv2)

        with tf.name_scope('pool2'):
            h_pool2 = max_pool_2x2(h_conv2, 'pool2')

        with tf.name_scope('fc1'):
            flattened_shape = (self.width >> 2) * (self.height >> 2) * self.conv2_features
            h_pool2_flat = tf.reshape(h_pool2, [-1, flattened_shape])

            W_fc1 = variable([flattened_shape, self.fc_out], 'w3')
            b_fc1 = variable([self.fc_out], 'b3')
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        with tf.name_scope('fc2'):
            W_fc2 = variable([self.fc_out, self.output_features], 'w4')
            b_fc2 = variable([self.output_features], 'b4')
            y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        return y_conv, keep_prob


class Policy(ConvolutionalNetwork):
    def __init__(self, width, height):
        super().__init__(width, height, conv1_features=16, conv2_features=32, fc_out=100)
        self.y_ = self.y_conv
        self.actions = tf.placeholder(tf.int32, name='input_a')
        self.rewards = tf.placeholder(tf.float32, name='input_r')

        self.opt, self.loss = self.optimiser()
        self.act = self.action()
        self.debug_all_var = tf.trainable_variables()

    def action(self):
        with tf.name_scope('output'):
            logits = tf.log(self.y_ / (1 - self.y_))
            return tf.squeeze(tf.multinomial(logits, 1), name='action')

    def optimiser(self):
        with tf.name_scope('loss'):
            indices = tf.range(0, tf.shape(self.y_)[0]) * tf.shape(self.y_)[1] + self.actions
            chosen_action_probs = tf.gather(tf.reshape(self.y_, [-1]), indices)
            loss = -tf.reduce_mean(tf.log(chosen_action_probs) * self.rewards)
        with tf.name_scope('adam'):
            optimiser = tf.train.AdamOptimizer(learning_rate=1e-4)
            return optimiser.minimize(loss, name='opt'), loss


if __name__ == '__main__':
    import numpy as np
    from tqdm import trange

    N = 239
    tf.reset_default_graph()
    agent = Policy(40, 80)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        actions_before_training = []
        sess.run(tf.global_variables_initializer())
        frames = [(np.random.randint(1, 255, 40 * 80) - 255.0) / 255.0 for _ in range(N)]

        for frame in frames:
            action = sess.run(agent.act, feed_dict={agent.input: [frame], agent.keep_prob: 1.0})
            actions_before_training.append(action)
            print(action)

        rewards = np.random.choice([-1.0, 0.0, 1.0], N, p=[0.1, 0.5, 0.4])
        actions = np.random.choice([0, 1], N, p=[0.9, 0.1])
        print('vars:', [var.eval() for var in agent.debug_all_var])

        print('-' * 25 + 'UPDATE' + '-' * 25)
        for _ in trange(25):
            sess.run(agent.opt, feed_dict={
                agent.input: frames,
                agent.rewards: rewards,
                agent.actions: actions,
                agent.keep_prob: 0.8
            })

        print('vars:', [var.eval() for var in agent.debug_all_var])
        for frame, before, data, reward in zip(frames, actions_before_training, actions, rewards):
            action = sess.run(agent.act, feed_dict={agent.input: [frame], agent.keep_prob: 1.0})
            if action != before:
                print('!!!!', end='')
            else:
                print('    ', end='')
            print('before:', before, 'data:', data, 'model:', action, 'reward:', reward)
