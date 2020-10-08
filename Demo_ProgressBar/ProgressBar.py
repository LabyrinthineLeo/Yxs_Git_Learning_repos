# coding  : utf-8
# 进度条实现
# @Author : Labyrinthine Leo
# @Time   : 2020.10.05

import requests
from requests.exceptions import RequestException
import tkinter as tk
from tkinter import *
from tkinter import StringVar
import tkinter.messagebox as ms
import time
import threading
import ctypes
import inspect
import os

# 全局变量
overkey = [0]

def main_fun(key_list):
    """
    主功能函数即识别函数(这里使用下载图片作为测试函数)
    """
    # captcha_url = "https://my.cnki.net/elibregister/CheckCode.aspx"
    captcha_url = "https://o.cnki.net/Register/CheckCode.aspx"
    headers = {
        'User-Agent':'Mozilla/5.0 (Windows NT 6.1; rv:2.0.2) Gecko/20100101 Firefox/4.0.1'
        # 'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3100.0 Safari/537.36'
    }
    for i in range(200):
        print("download {} picture".format(i))
        try:
            res = requests.get(captcha_url, headers=headers)
            if res.status_code == 200:
                res = res.content
            else:
                res = None
        except RequestException:
            res = None
        if res is not None:
            with open('./Images_Set/{}.gif'.format(i),'wb') as gif:
                gif.write(res)
        else:
            continue
    # 执行完毕设置为1
    key_list.append(1)

class ProgressBar:
    def __init__(self):
        # 创建主窗口
        self.window = tk.Tk()
        self.window.title('ProgressBar')
        self.window.geometry('650x150+500+440')
        # 设置关闭窗口监听事件
        self.window.protocol('WM_DELETE_WINDOW',self.closeWindow) 
        # 设置背景图
        self.photo=PhotoImage(file=r"./background.gif")
        self.label=Label(self.window,image=self.photo)  #图片
        self.label.pack()

         
        # 设置下载进度条
        tk.Label(self.window, text='识别进度:', ).place(x=50, y=60)
        # 创建一个背景白色的矩形
        self.canvas = tk.Canvas(self.window, width=465, height=22, bg="white")

        self.canvas.place(x=110, y=60)

        # progress()
        self.var = StringVar()
        self.var.set("开始识别")
        self.btn_start = tk.Button(self.window, textvariable=self.var, command=self.progress)
        self.btn_start.place(x=585, y=60) # 启动进度条
         
        self.window.mainloop()

    def progress(self):
        global overkey
        self.thread_a = threading.Thread(target=main_fun,args=(overkey,))
        self.thread_a.start()
        # print("thread_b ... ")
        self.btn_start.config(state='disable') # 设置按钮只能点击一次
        # 填充进度条
        fill_line = self.canvas.create_rectangle(1.5, 1.5, 0, 23, width=0, fill="#6D5D72")
        x = 500  #
        n = 465 / x  # 465是矩形填充满的次数
        k = 100 / x
        for i in range(x-6):
            n += 465 / x
            k += 100 / x
            self.canvas.coords(fill_line, (0, 0, n, 60))
            if k>100:
                self.var.set("100.0%")
            else:
                self.var.set(str(round(k,1))+"%")
            if i<200:
                time.sleep(0.02)  # 控制进度条流动的速度
            elif i<400:
                time.sleep(0.1)
            else:
                time.sleep(0.03)
            self.window.update()

        # overkey = 1
        while True:
            if overkey[-1]==1:
                # 填充进度条
                fill_line = self.canvas.create_rectangle(1.5, 1.5, 465, 23, width=0, fill="#6D5D72")
                self.var.set("100.0%")
                self.window.update()
                break
        time.sleep(1)
        ms.showinfo('Message','识别完成！')
        self.window.destroy()
        # self.stop_thread(self.thread_a)

        # 清空进度条
        # fill_line = canvas.create_rectangle(1.5, 1.5, 0, 23, width=0, fill="white")
        # x = 500  # 未知变量，可更改
        # n = 465 / x  # 465是矩形填充满的次数
        # for t in range(x):
        #     n = n + 465 / x
        #     # 以矩形的长度作为变量值更新
        #     canvas.coords(fill_line, (0, 0, n, 60))
        #     window.update()
        #     time.sleep(0)  # 时间为0，即飞速清空进度条
    
    def closeWindow(self):
        if ms.askokcancel("Warning",'Close the Window?'):
            os._exit(0) # 停止所有程序
            self.stop_thread(self.thread_a)
            self.window.destroy()


    def _async_raise(self, tid, exctype):
        """raises the exception, performs cleanup if needed"""
        tid = ctypes.c_long(tid)
        if not inspect.isclass(exctype):
            exctype = type(exctype)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
        if res == 0:
            raise ValueError("invalid thread id")
        elif res != 1:
            # """if it returns a number greater than one, you're in trouble,
            # and you should call it again with exc=NULL to revert the effect"""
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")
 
    def stop_thread(self,thread): # 手动kill线程
        self._async_raise(thread.ident, SystemExit)


if __name__ == '__main__':
    pb = ProgressBar()