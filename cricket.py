import numpy as np
import pyautogui
import win32gui
import win32ui
import win32con
import win32api
import os
from time import sleep


class CricketBot:

    region_loc = os.path.join('res', 'region.PNG')
    
    def __init__(self, region=None):
        self.red = (34,34,187,255)
        self.region = region
        self.click = None
        
        
    def set_pos(self):
        if self.region:
            self.click = (self.region[0], self.region[1])
            return
        
        self.region = self._locate(self.region_loc)
            
        if self.region:
            self.click = (self.region[0], self.region[1])
            print(self.click)
            pyautogui.click(self.click)
        else:
            raise Exception('game area not found')
            
    def _locate(self, pic):
        loc = pyautogui.locateOnScreen(pic)
        return (loc[0], loc[1], loc[0] + loc[2], loc[1] + loc[3]) if loc else None
        

    def grab_screen(self):
        """
        an order of magnitude improvement (compared to self.grab_simple)
        """
        left,top,x2,y2 = self.region
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
        img.shape = (height,width,4)

        memdc.DeleteDC()
        win32gui.DeleteObject(bmp.GetHandle())
        win32gui.ReleaseDC(win32gui.GetDesktopWindow(), hwindc)  
        return img
        
        
    def grab_simple(self):
        """
        this function was too slow, the bot wasn't able to hit the target in time
        """
        img = pyautogui.screenshot(region=self.region)
        mtx = np.array(img)
        return mtx
        

    def ball_detected(self, mtx):
        for x in mtx:
            for y in x:
                if y[0] == self.red[0] and y[1] == self.red[1] and y[2] == self.red[2]:
                    return True


    def start(self):
        self.set_pos()
        print("bot... running...")
        i = 0
        while True:
            i += 1
            if i % 250 == 0:
                print(i)
                        
            mtx = self.grab_screen()
            if self.ball_detected(mtx):
                pyautogui.click(self.click)
                
if __name__ == "__main__":
    coord = (497, 382)  # the eye of the cricket 
    offset = 50
    region = (coord[0], coord[1], coord[0]+offset, coord[1]+offset)
    bot = CricketBot(region)
    bot.start()
