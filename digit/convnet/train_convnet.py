import os
from os import path
import numpy as np
import tensorflow as tf
from tqdm import trange
from digit.data import sample_validation_data
from digit.data import read_data
from digit.convnet.conv_net import ConvNet


cwd = os.getcwd()
checkpoints_dir = 'convnet_checkpoint/'
training = path.join(cwd, 'training')


def create_checkpoints_dir():
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)


def train(model):
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for _ in trange(20000):
            batch = read_data(training, 50)
            model.opt.run(feed_dict={model.x: batch[0],
                                     model.y_: batch[1],
                                     model.keep_prob: .8})

        new_image = sample_validation_data()
        prediction = sess.run(model.y_conv, {model.x: new_image, model.keep_prob: 1.0})
        print('identified as:', np.argmax(prediction))

        saver.save(sess, './' + checkpoints_dir + 'digit.ckpt')

if __name__ == '__main__':
    create_checkpoints_dir()
    tf.reset_default_graph()

    net = ConvNet()
    train(net)
