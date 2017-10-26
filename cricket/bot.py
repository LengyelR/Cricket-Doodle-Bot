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
from os import path
from PIL import Image

cwd = os.getcwd()
checkpoints_dir = 'cricket/'
training = path.join(cwd, 'training')

training_log = r'C:\tmp\training_log'


def create_checkpoints_dir():
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)


def save_episode(frames, action, reward, counter, episode):
    for idx, mtx in enumerate(frames):
        plt.imsave(arr=mtx,
                   fname=path.join(training_log,
                                   'e' + str(episode) +
                                   'c' + str(counter) +
                                   'i' + str(idx) +
                                   '_action' + str(action) +
                                   "_reward" + str(reward) + '.png'),
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
        res = []
        for mtx in frames:
            im = Image.fromarray(mtx)
            mtx = np.array(im.convert('L'))
            mtx = (mtx - 255.0) / 255.0
            res.extend(mtx.reshape(23*25))
        return res

    @staticmethod
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

        if r == 6:
            return 2.0

        elif r == 0:
            return 1.0

        else:
            return -1.0

    def train_with_bot(self):
        create_checkpoints_dir()
        tf.reset_default_graph()
        cn = model.LoadModel(self.model_directory, 'fc2/y_conv_model', 'digit')
        agent = model.Policy()

        batch_size = 1
        game_counter = 1
        s_history, a_history, r_history = [], [], []

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("bot... running...")
            print('vars:', [var.eval() for var in agent.debug_all_var])

            while True:
                mtx = self.grab_screen((self.zero[0], self.zero[1],
                                        self.zero[0] + 538, self.zero[1] + 300))
                score_mtx = mtx[22:45, 253:285]
                incoming_ball_mtx = mtx[187:210, 254:279]
                possible_end = mtx[270:287, 210:230]

                if np.sum(possible_end[0:, 0, 0]) > 4250 \
                        or (self.previous_digit == 9 and self.current_digit == 9):
                    print('GAME OVER')
                    self.previous_digit = 0
                    self.current_digit = 0
                    time.sleep(30)
                    pyautogui.click(self.click)
                    time.sleep(2)

                if self.ball_detected(incoming_ball_mtx):
                    pyautogui.click(self.click)
                    print('-' * 25)
                    print('GAME:', game_counter-1)
                    batch_size += 1
                    game_counter += 1
                    digit_mtx = self.preprocess(score_mtx)

                    res = cn.run(digit_mtx.reshape(1, 196))
                    digit = np.argmax(res)
                    self.previous_digit = self.current_digit
                    self.current_digit = digit
                    r = self.reward()

                    # mimic the bot's behaviour
                    s_history.append(self.preprocess_ball([incoming_ball_mtx]))
                    a_history.append(1)
                    r_history.append(1.0)

                    print('DIGIT:', digit)
                    print('REWARD:', r)
                    if game_counter % 25 == 0:
                        print('----------SAVE----------')
                        saver.save(sess, './' + checkpoints_dir + 'cricket_bot.ckpt')
                else:
                    s_history.append(self.preprocess_ball([incoming_ball_mtx]))
                    a_history.append(0)
                    r_history.append(1.0)

                if batch_size % 10 == 0:
                    print('---------UPDATE---------')
                    feed_dict = {agent.input: s_history,
                                 agent.actions: a_history,
                                 agent.rewards: r_history}
                    agent.opt.run(feed_dict=feed_dict)
                    loss = agent.loss.eval(feed_dict=feed_dict)
                    print('vars:', [var.eval() for var in agent.debug_all_var])
                    print('loss:', loss)
                    with open(path.join(training_log, 'log.txt'), 'a') as logf:
                        logf.writelines(str(loss) + '\n')

                    s_history.clear()
                    a_history.clear()
                    r_history.clear()
                    batch_size = 1

    def run_policy(self):
        policy = model.LoadModel('cricket', 'output/action', 'cricket_bot', dropout=False)
        while True:
            mtx = self.grab_screen((self.zero[0], self.zero[1],
                                    self.zero[0] + 538, self.zero[1] + 300))
            incoming_ball_mtx = mtx[187:210, 254:279]
            possible_end = mtx[270:287, 210:230]

            state = self.preprocess_ball([incoming_ball_mtx])
            action = policy.run([state])
            if action:
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
    bot.train_with_bot()
    # bot.run_policy()
