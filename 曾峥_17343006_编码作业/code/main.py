# -*- coding: utf-8 -*-
"""
运行本项目需要安装的库：
    keras 2.2.4
    PyQt5 5.11.3
    pandas 0.24.2
    scikit-learn 0.21.2
    tensorflow 1.13.1
    imutils 0.5.2
    opencv-python 4.10.25
    matplotlib 3.2.1  # 注意：此依赖包为第二版新增，请注意安装

点击运行主程序Main.py
"""

import warnings
import os
# 忽略警告

from ui_mainWindow import MainWindow
from sys import argv, exit
from PyQt5.QtWidgets import QApplication

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings(action='ignore')

if __name__ == '__main__':
    app = QApplication(argv)
    window = MainWindow()
    window.show()
    exit(app.exec_())
