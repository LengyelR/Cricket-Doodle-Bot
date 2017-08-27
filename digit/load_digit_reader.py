import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from digit.my_data import get_new_image
from digit.train_digit_reader import read_data

data_path = r'C:\tmp\images\game7\all_data.txt'
checkpoints_dir = 'cricket_checkpoints_drop'
# checkpoints_dir = 'checkpoints'
# checkpoints_dir = 'checkpoints_drop'


def main(_):
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        saver = tf.train.import_meta_graph('./' + checkpoints_dir + '/digit.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./' + checkpoints_dir))

        y_conv = loaded_graph.get_tensor_by_name("fc2/y_conv_model:0")
        x_tensor = loaded_graph.get_tensor_by_name("input_x:0")
        keep_prob = loaded_graph.get_tensor_by_name("dropout/keep_prob:0")
        image = get_new_image()

        res = sess.run(y_conv, feed_dict={x_tensor: image, keep_prob: 1.0})
        prediction = np.argmax(res)
        print(prediction)

        results = []
        counter = 0
        with open(data_path, 'r') as f:
            raw_data = f.readlines()

        for xs, labels in read_data(raw_data):
            for x, label in zip(xs, labels):
                my_pred = sess.run(y_conv, feed_dict={x_tensor: x.reshape(1, 784), keep_prob: 1.0})
                correct_label = np.argmax(label)
                my_label = np.argmax(my_pred)
                results.append(my_label == correct_label)
                counter += 1
                if my_label != correct_label:
                    plt.imsave(arr=x.reshape((28, 28)),
                               fname='C:\\tmp\images\\test\\' + str(counter) + "pred_" + str(my_label) + '.png',
                               cmap='gray')

        all = len(results)
        correct = float(sum(results))
        print('all', all)
        print('correct', correct)
        print('accuracy:', correct/all)

if __name__ == '__main__':
    tf.app.run()