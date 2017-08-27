import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from digit.conv_net import ConvNet
from digit.my_data import get_new_image

checkpoints_dir = 'cricket_checkpoints_drop/'
data_path = r'C:\tmp\images\game7\all_data.txt'


def create_checkpoints_dir():
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)


def read_data(raw):
    data_batch = []
    label_batch = []
    for data in raw:
        if data[0].lower() == 'x':
            continue

        digit = int(data[0])
        label = np.zeros(10)
        label[digit] = 1
        raw = data[2:].rstrip().split(' ')
        a = np.array([float(x) for x in raw])

        label_batch.append(label)
        data_batch.append(a)

        if len(data_batch) == 50:
            yield data_batch, label_batch
            data_batch = []
            label_batch = []


def train(net, raw_data):
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for batch in read_data(raw_data):
            net.opt.run(feed_dict={net.x: batch[0],
                                   net.y_: batch[1],
                                   net.keep_prob: .8})

        new_image = get_new_image()
        prediction = sess.run(net.y_conv, {net.x: new_image, net.keep_prob: 1.0})
        print('identified as:', np.argmax(prediction))

        saver.save(sess, './' + checkpoints_dir + 'digit.ckpt')

if __name__ == '__main__':
    create_checkpoints_dir()
    tf.reset_default_graph()

    with open(data_path, 'r') as f:
        raw_data = f.readlines()
    np.random.shuffle(raw_data)
    net = ConvNet()

    train(net, raw_data)
