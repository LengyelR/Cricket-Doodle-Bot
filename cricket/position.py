import pyautogui
import time
from sys import stdout


def current_pos():
    while True:
        time.sleep(0.75)
        pos = pyautogui.position()
        stdout.write("\r" + str(pos))
        stdout.flush()

if __name__ == "__main__":
    current_pos()
