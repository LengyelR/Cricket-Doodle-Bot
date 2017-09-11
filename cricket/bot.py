import pyautogui
import win32gui
import win32ui
import win32con
import os
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def preprocess(mtx):
    im = Image.fromarray(mtx)
    im = im.convert('L')

    area = (20, 5, 34, 18)
    cropped_img = im.crop(area)

    mtx = np.array(cropped_img)
    mtx[mtx < 144] = 0
    mtx[:, 0] = 0

    for index, col in enumerate(mtx.T):
        if np.count_nonzero(col) in (1, 2):
            mtx[:, index] = np.zeros(13)

    non_empty_columns = np.where(mtx.max(axis=0) > 0)[0]
    non_empty_rows = np.where(mtx.max(axis=1) > 0)[0]

    border = (min(non_empty_rows), max(non_empty_rows)+1,
              min(non_empty_columns), max(non_empty_columns)+1)

    cut = mtx[border[0]:border[1], border[2]:border[3]]
    row = cut.shape[0]
    col = cut.shape[1]

    row_start = (14 - row) // 2
    row_end = row_start + row
    col_start = (14 - col) // 2
    col_end = col_start + col

    paste = np.zeros((14, 14))
    paste[row_start:row_end, col_start:col_end] = cut

    return paste


class CricketBot:
    region_loc = os.path.join('res', 'region.PNG')

    def __init__(self, zero, image_path, nwc):
        self.red = (34, 34, 187, 255)
        self.zero = zero
        self.click = (zero[0]+270, zero[1]+277)
        self.image_path = image_path
        self.network_checkpoint = nwc
        self.previous_digit = 0
        self.current_digit = 0

    @staticmethod
    def grab_screen(area):
        left, top, x2, y2 = area
        width = x2 - left + 1
        height = y2 - top + 1

        hwin = win32gui.GetDesktopWindow()
        hwindc = win32gui.GetWindowDC(hwin)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, width, height)
        memdc.SelectObject(bmp)

        memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
        signedIntsArray = bmp.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (height, width, 4)

        memdc.DeleteDC()
        win32gui.DeleteObject(bmp.GetHandle())
        win32gui.ReleaseDC(win32gui.GetDesktopWindow(), hwindc)
        return img

    @staticmethod
    def read_score(score_mtx):
        image = Image.fromarray(score_mtx)
        bw = np.array(image.convert('L'))
        bw[bw < 150] = 0

        non_empty_columns = np.where(bw.max(axis=0) > 0)[0]
        non_empty_rows = np.where(bw.max(axis=1) > 0)[0]

        border = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
        cut = bw[border[0]:border[1], border[2]:border[3]]
        row = cut.shape[0]
        col = cut.shape[1]

        row_start = (14 - row) // 2
        row_end = row_start + row
        col_start = (14 - col) // 2
        col_end = col_start + col

        paste = np.zeros((14, 14))
        paste[row_start:row_end, col_start:col_end] = cut
        image = Image.fromarray(paste)
        image = image.resize((28, 28))
        mtx = np.array(image)
        return mtx.reshape(1, 784)

    def ball_detected(self, mtx):
        for x in mtx:
            for y in x:
                if y[0] == self.red[0] and y[1] == self.red[1] and y[2] == self.red[2]:
                    return True

    def reward(self):
        r = (self.current_digit - self.previous_digit % 10)
        return r if 0 <= r else r + 10

    def start(self):
        print("bot... running...")
        counter = 0
        loaded_graph = tf.Graph()

        with tf.Session(graph=loaded_graph) as sess:
            saver = tf.train.import_meta_graph(self.network_checkpoint + '/digit.ckpt.meta')
            saver.restore(sess, tf.train.latest_checkpoint(self.network_checkpoint))

            y_conv = loaded_graph.get_tensor_by_name("fc2/y_conv_model:0")
            x_tensor = loaded_graph.get_tensor_by_name("input_x:0")
            keep_prob_tensor = loaded_graph.get_tensor_by_name("dropout/keep_prob:0")

            while True:
                mtx = self.grab_screen((self.zero[0], self.zero[1],
                                        self.zero[0]+538, self.zero[1]+300))
                score_mtx = mtx[22:45, 253:285]
                incoming_ball_mtx = mtx[187:210, 254:279]
                possible_end = mtx[270:287, 210:230]

                if np.sum(possible_end[0:, 0, 0]) > 4250:
                    print('GAME OVER')
                    self.previous_digit = 0
                    self.current_digit = 0
                    time.sleep(5)
                    pyautogui.click(self.click)
                    time.sleep(2)

                if self.ball_detected(incoming_ball_mtx):
                    pyautogui.click(self.click)

                    print('CLICK:', counter)
                    counter += 1
                    digit_mtx = preprocess(score_mtx)

                    res = sess.run(y_conv, feed_dict={x_tensor: digit_mtx.reshape(1, 196), keep_prob_tensor: 1.0})
                    digit = np.argmax(res)

                    plt.imsave(arr=digit_mtx,
                               fname=f"{self.image_path}{digit}_{counter}.png",
                               cmap='gray')
                    self.previous_digit = self.current_digit
                    self.current_digit = digit
                    print(digit)
                    print('REWARD:', self.reward())

if __name__ == "__main__":
    checkpoint = '../digit/convnet/convnet_checkpoint'
    log = 'C:\\tmp\images\\game9\\'
    f = lambda x: Image.fromarray(x).show()

    top_left_game = (253, 208)

    bot = CricketBot(top_left_game, log, checkpoint)
    bot.start()
