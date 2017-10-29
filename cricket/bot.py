import os
import pyautogui
import win32gui
import win32ui
import win32con
import time
import numpy as np
import cricket.model as model
import tensorflow as tf
import matplotlib.pyplot as plt
import collections
from os import path
from PIL import Image

cwd = os.getcwd()
checkpoints_dir = 'cricket/'
training = path.join(cwd, 'training')

training_log = r'C:\tmp\training_log'


def create_checkpoints_dir():
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)


def save_episode(frames, reward, game):
    for idx, mtx in enumerate(frames):
        mtx = np.array(mtx)
        plt.imsave(arr=mtx.reshape(40, 40),
                   fname=path.join(training_log,
                                   'R' + str(reward) +
                                   '_G' + str(game) +
                                   '_I' + str(idx) + '.png'),
                   cmap='gray')


class CricketBot:
    def __init__(self, zero, image_path, model_dir):
        self.red = (34, 34, 187, 255)
        self.zero = zero
        self.click = (zero[0] + 270, zero[1] + 277)
        self.image_path = image_path
        self.model_directory = model_dir
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
    def preprocess_ball(frames):
        w = frames[0].shape[0]
        h = frames[0].shape[1]
        res = np.zeros((w, h))
        for mtx in frames:
            mtx = mtx[:, :, 1]
            mtx[mtx > 35] = 255
            mtx = (mtx - 255.0) / 255.0
            res += mtx

        res = res.reshape(w*h)
        return res

    @staticmethod
    def preprocess_digits(mtx):
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

        border = (min(non_empty_rows), max(non_empty_rows) + 1,
                  min(non_empty_columns), max(non_empty_columns) + 1)

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

    def ball_detected(self, mtx):
        for x in mtx:
            for y in x:
                if y[0] == self.red[0] and y[1] == self.red[1] and y[2] == self.red[2]:
                    return True

    def reward(self):
        r = (self.current_digit - self.previous_digit) % 10
        return r

    def train_with_bot(self):
        create_checkpoints_dir()
        tf.reset_default_graph()
        cn = model.LoadModel(self.model_directory, 'fc2/y_conv_model', 'digit')
        agent = model.Policy(40, 40)

        batch_size = 1
        game_counter = 1
        s_history, a_history, r_history = [], [], []
        frames = collections.deque(maxlen=5)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("bot... running...")

            while True:
                mtx = self.grab_screen((self.zero[0], self.zero[1],
                                        self.zero[0] + 538, self.zero[1] + 300))
                score_mtx = mtx[22:45, 253:285]
                incoming_ball_mtx = mtx[187:210, 254:279]
                big_ball_mtx = mtx[160:200, 255:295]
                possible_end = mtx[270:287, 210:230]
                frames.append(big_ball_mtx)

                if np.sum(possible_end[0:, 0, 0]) > 4250 \
                        or (self.previous_digit == 9 and self.current_digit == 9):
                    print('GAME OVER')
                    self.previous_digit = 0
                    self.current_digit = 0
                    s_history.clear()
                    a_history.clear()
                    r_history.clear()
                    frames.clear()

                    time.sleep(30)
                    pyautogui.click(self.click)
                    time.sleep(2)

                if self.ball_detected(incoming_ball_mtx):
                    pyautogui.click(self.click)
                    print('-' * 25)
                    print('GAME:', game_counter-1)
                    batch_size += 1
                    game_counter += 1
                    digit_mtx = self.preprocess_digits(score_mtx)

                    res = cn.run(digit_mtx.reshape(1, 196))
                    digit = np.argmax(res)
                    self.previous_digit = self.current_digit
                    self.current_digit = digit
                    r = self.reward()

                    # mimic the bot's behaviour
                    if len(frames) == 5:
                        s_history.append(self.preprocess_ball(frames))
                        a_history.append(1)
                        r_history.append(1.0)

                    print('DIGIT:', digit)
                    print('REWARD:', r)
                    if game_counter % 50 == 0:
                        print('----------SAVE----------')
                        saver.save(sess, './' + checkpoints_dir + 'cricket_bot.ckpt')
                else:
                    if len(frames) == 5:
                        s_history.append(self.preprocess_ball(frames))
                        a_history.append(0)
                        r_history.append(1.0)

                if batch_size % 5 == 0:
                    update = model.UpdateThread(sess, agent,
                                                s_history, a_history, r_history,
                                                game_counter, training_log)
                    update.start()

                    s_history = []
                    a_history = []
                    r_history = []
                    batch_size = 1

    def run_policy(self):
        policy = model.LoadModel('cricket', 'output/action', 'cricket_bot', dropout=True)
        frames = collections.deque(maxlen=5)
        for _ in range(5):
            mtx = self.grab_screen((self.zero[0], self.zero[1],
                                    self.zero[0] + 538, self.zero[1] + 300))
            big_ball_mtx = mtx[160:200, 255:295]
            frames.append(big_ball_mtx)

        while True:
            mtx = self.grab_screen((self.zero[0], self.zero[1],
                                    self.zero[0] + 538, self.zero[1] + 300))
            big_ball_mtx = mtx[160:200, 255:295]
            possible_end = mtx[270:287, 210:230]

            frames.append(big_ball_mtx)
            state = self.preprocess_ball(frames)
            action = policy.run([state])
            if action:
                pyautogui.click(self.click)
                print('CLICKED')

            if np.sum(possible_end[0:, 0, 0]) > 4250 \
                    or (self.previous_digit == 9 and self.current_digit == 9):
                print('GAME OVER')
                self.previous_digit = 0
                self.current_digit = 0

                time.sleep(30)
                pyautogui.click(self.click)
                time.sleep(2)


if __name__ == "__main__":
    checkpoint = '../digit/convnet/convnet_checkpoint'
    log = 'C:\\tmp\images\\game9\\'
    f = lambda x: Image.fromarray(x).show()

    top_left_game = (253, 208)
    # top_left_game = (253, 186)
    bot = CricketBot(top_left_game, log, checkpoint)
    # bot.train_with_bot()
    bot.run_policy()
