import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from os import path
from digit.data import read_data

checkpoints_dir = 'convnet_checkpoint'
cwd = os.getcwd()
validate = path.join(cwd, 'validate')
error_path = path.join(cwd, 'errors')


def main(_):
    if not os.path.exists(error_path):
        os.makedirs(error_path)

    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        saver = tf.train.import_meta_graph('./' + checkpoints_dir + '/digit.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./' + checkpoints_dir))

        y_conv = loaded_graph.get_tensor_by_name("fc2/y_conv_model:0")
        x_tensor = loaded_graph.get_tensor_by_name("input_x:0")
        keep_prob = loaded_graph.get_tensor_by_name("dropout/keep_prob:0")

        results = []
        counter = 0
        xs, labels = read_data(validate)

        for x, label in zip(xs, labels):
            my_pred = sess.run(y_conv, feed_dict={x_tensor: x.reshape(1, 196), keep_prob: 1.0})
            correct_label = np.argmax(label)
            my_label = np.argmax(my_pred)
            results.append(my_label == correct_label)
            counter += 1
            if my_label != correct_label:
                plt.imsave(arr=x.reshape((14, 14)),
                           fname=path.join(error_path,
                                           str(counter) + 'label_' + str(correct_label)
                                           + "pred_" + str(my_label) + '.png'),
                           cmap='gray')

        all = len(results)
        correct = float(sum(results))
        print('all', all)
        print('correct', correct)
        print('accuracy:', correct/all)

if __name__ == '__main__':
    tf.app.run()