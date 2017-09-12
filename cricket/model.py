import tensorflow as tf


class LoadConvNet():
    def __init__(self, loc, output, name, input_layer='input_x'):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.input = input_layer + ':0'

        with self.graph.as_default():
            saver = tf.train.import_meta_graph(loc + '/' + name + '.ckpt.meta', clear_devices=True)
            saver.restore(self.sess, tf.train.latest_checkpoint(loc))

            self.output = self.graph.get_tensor_by_name(output + ':0')

    def run(self, data):
        return self.sess.run(self.output, feed_dict={self.input: data, "dropout/keep_prob:0": 1.0})