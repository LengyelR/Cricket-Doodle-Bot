import numpy as np
import matplotlib.pyplot as plt
import os
from os import path
from PIL import Image

img = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 108, 145, 109, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 152, 253, 255, 254, 143, 0, 0, 0, 0,
                0, 0, 0, 0, 108, 252, 250, 232, 251, 252, 0, 0, 0, 0,
                0, 0, 0, 0, 175, 249, 118, 0, 140, 254, 153, 0, 0, 0,
                0, 0, 0, 0, 237, 203, 0, 0, 0, 229, 189, 0, 0, 0,
                0, 0, 0, 0, 255, 185, 0, 0, 0, 216, 195, 0, 0, 0,
                0, 0, 0, 0, 251, 211, 0, 0, 0, 242, 180, 0, 0, 0,
                0, 0, 0, 0, 208, 253, 152, 110, 201, 255, 143, 0, 0, 0,
                0, 0, 0, 0, 140, 255, 255, 254, 255, 244, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 209, 255, 255, 250, 132, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 141, 164, 106, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


def get_new_image():
    reshaped = img.reshape(14, 14)
    image = Image.fromarray(reshaped)
    image = image.resize((28, 28))
    image = np.array(image)
    return image.reshape(1, 784)


def sample_validation_data():
    cwd = os.getcwd()
    conv_net_dir = path.join(cwd, 'convnet')
    directory = path.join(conv_net_dir, 'validate')

    dataset = os.listdir(directory)
    np.random.shuffle(dataset)

    im = Image.open(path.join(directory, dataset[0]))
    im = im.convert('L')
    mtx = np.array(im)
    print(mtx)
    array = mtx.reshape(1, 196)
    return array


def read_data(directory, batch_size=None):
    data_batch = []
    label_batch = []
    dataset = os.listdir(directory)
    np.random.shuffle(dataset)

    batch_size = len(dataset) if batch_size is None else batch_size

    for pic in dataset:
        im = Image.open(path.join(directory, pic))
        im = im.convert('L')
        mtx = np.array(im)
        array = mtx.reshape(196)

        digit = int(pic.split('_')[0])
        label = np.zeros(10)
        label[digit] = 1

        label_batch.append(label)
        data_batch.append(array)

        if len(data_batch) == batch_size:
            return data_batch, label_batch


if __name__ == '__main__':
    im = get_new_image()
    print(im)

    pic = im.reshape((28,28))
    plt.imshow(pic, cmap='gray')
    plt.show()
