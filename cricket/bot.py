import pyautogui
import win32gui
import win32ui
import win32con
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cricket.utils as utils
import time
from PIL import Image


class CricketBot:
    region_loc = os.path.join('res', 'region.PNG')

    def __init__(self, region, score, image_path, nwc):
        self.red = (34, 34, 187, 255)
        self.region = region
        self.score_region = score
        self.click = (region[0], region[1])
        self.image_path = image_path
        self.network_checkpoint = nwc

    def grab_screen(self, area=None):
        left, top, x2, y2 = self.score_region if area is None else area
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

    def read_score(self):
        score_pic = bot.grab_screen()
        image = Image.fromarray(score_pic)
        image = image.resize((28, 28))

        mtx = np.array(image)
        bw = mtx[:, :, 2]
        bw[bw < 150] = 0
        bw[:, 0:4] = 0
        bw[:, -3:] = 0
        bw[0, :] = 0
        bw[-1:, :] = 0

        non_empty_columns = np.where(bw.max(axis=0) > 0)[0]
        non_empty_rows = np.where(bw.max(axis=1) > 0)[0]

        if len(non_empty_columns) == 0 \
                or len(non_empty_rows) == 0\
                or np.sum(bw[8:15, 8:15]) > 200*49 \
                or np.sum(bw) < 1000:
            time.sleep(0.05)
            return None

        border = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
        cut = bw[border[0]:border[1], border[2]:border[3]]
        row = cut.shape[0]
        col = cut.shape[1]

        row_start = (28 - row) // 2
        row_end = row_start + row
        col_start = (28 - col) // 2
        col_end = col_start + col

        paste = np.zeros((28, 28))
        paste[row_start:row_end, col_start:col_end] = cut
        return paste.reshape(1, 784)

    def ball_detected(self, mtx):
        for x in mtx:
            for y in x:
                if y[0] == self.red[0] and y[1] == self.red[1] and y[2] == self.red[2]:
                    return True

    def start(self):
        print("bot... running...")
        i = 0
        counter = 0
        click_counter = 0
        ball_was_hit = False
        possible_end_of_ep = utils.make_square((526, 313), 10)

        loaded_graph = tf.Graph()

        with tf.Session(graph=loaded_graph) as sess:
            saver = tf.train.import_meta_graph('../digit/' + checkpoint + '/digit.ckpt.meta')
            saver.restore(sess, tf.train.latest_checkpoint('../digit/' + checkpoint))

            y_conv = loaded_graph.get_tensor_by_name("fc2/y_conv_model:0")
            x_tensor = loaded_graph.get_tensor_by_name("input_x:0")
            keep_prob_tensor = loaded_graph.get_tensor_by_name("dropout/keep_prob:0")
            while True:
                i += 1
                if i % 250 == 0:
                    print('frames:', i)

                mtx = self.grab_screen(self.region)
                new_throw = self.grab_screen(possible_end_of_ep)

                if self.ball_detected(mtx):
                    pyautogui.click(self.click)
                    click_counter += 1
                    print('CLICK:', click_counter)
                    ball_was_hit = True

                if ball_was_hit and self.ball_detected(new_throw):
                    counter += 1
                    score = self.read_score()
                    if score is None:
                        print('UNKNOWN:', counter)
                        ball_was_hit = False
                        continue

                    res = sess.run(y_conv, feed_dict={x_tensor: score, keep_prob_tensor: 1.0})
                    prediction = np.argmax(res)

                    pairs = []
                    for i, s in enumerate(res[0]):
                        pairs.append((i, s))
                    top = sorted(pairs, key=lambda k: k[1], reverse=True)[:3]

                    s = ''
                    total = sum([k[1] for k in top])
                    for pair in top:
                        s += f'({pair[0]}, {round(pair[1]/total, 2)}) '
                    print(s)
                    conf = round(top[0][1]/total, 2)

                    utils.write_array(score[0], str(prediction), self.image_path + 'logs.txt')
                    utils.write_array(s, str(counter), self.image_path + 'cc.txt')
                    plt.imsave(arr=score.reshape((28, 28)),
                               fname=f"{self.image_path}{counter}_{conf}_{prediction}.png",
                               cmap='gray')
                    print('EPISODE:', counter)
                    ball_was_hit = False


if __name__ == "__main__":
    checkpoint = 'cricket_checkpoints_drop'
    log = 'C:\\tmp\images\\game8\\'

    top_left_game = (497, 382)
    top_left_score = (526, 235)
    game = utils.make_square(top_left_game, 50)
    score_region = utils.make_square(top_left_score, 12)

    bot = CricketBot(game, score_region, log, checkpoint)
    bot.start()
