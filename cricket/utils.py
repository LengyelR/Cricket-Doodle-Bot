import numpy as np
import matplotlib.pyplot as plt


def make_square(top_left_corner, offset):
    return top_left_corner[0], top_left_corner[1], top_left_corner[0] + offset, top_left_corner[1] + offset


def softmax(x, f=np.exp):
    z = f(x)
    return z / sum(z)


def show_image(img):
    pic = img.reshape((28, 28))
    plt.imshow(pic)
    plt.show()


def write_array(img, name, path):
    s = ''
    for px in img:
        s += str(px) + ' '
    s += '\n'
    with open(path, 'a') as f:
        f.writelines(name + "\t" + s)


def write(s, name, path):
    with open(path, 'a') as f:
        f.writelines(name + "\t" + s)
