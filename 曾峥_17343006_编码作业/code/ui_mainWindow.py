import sip
from PyQt5 import QtCore, QtGui, QtWidgets
from ui.window import Ui_Form  # 分离的UI界面
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QProgressBar, QTextBrowser  # 选择本地文件窗口
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMovie  # 显示.gif图像
from real_time_video_me import Emotion_Rec  # 核心调用，表情识别类
from os import getcwd  # 返回当前工作目录
import numpy as np
import cv2
import time
import sys

sys.path.append('../')
import qrc.resource  # dirName.fileName


class MainWindow(QtWidgets.QMainWindow, Ui_Form):
    # class MainWindow(object):
    def __init__(self, parent=None):
        # super(MainWindow, self).__init__(parent)
        QMainWindow.__init__(self)
        Ui_Form.__init__(self)

        self.EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",
                         "neutral"]
        self.path = getcwd()
        self.timer_camera = QtCore.QTimer()  # 定时器

        self.setupUi(self)
        self.retranslateUi(self)
        self.slot_init()  # 槽函数设置

        # 设置界面动画
        gif = QMovie(':/images/ui/scan.gif')
        self.label_face.setMovie(gif)
        gif.start()

        self.cap = cv2.VideoCapture()  # 屏幕画面对象
        self.CAM_NUM = 0  # 摄像头标号
        self.model_path = None  # 模型路径
        self.showResult()  # 结果条形码无有效数据

    def slot_init(self):  # 定义槽函数
        self.toolButton_model.clicked.connect(self.choose_model)
        self.toolButton_camera.clicked.connect(self.button_open_camera_click)
        self.timer_camera.timeout.connect(self.show_camera)
        self.toolButton_file.clicked.connect(self.choose_pic)

    def choose_model(self):
        # 选择训练好的模型文件
        self.timer_camera.stop()
        self.cap.release()
        self.label_face.clear()
        self.textBrowser_result.setText('None')
        self.textBrowser_time.setText('0 s')
        self.textEdit_camera.setText('Camera off')
        self.showResult()  # 条形图无数据

        # 调用文件选择对话框
        fileName_choose, filetype = QFileDialog.getOpenFileName(self,
                                                                "Select a file...", getcwd(),  # 起始路径
                                                                "Model File (*.hdf5)")  # 文件类型
        # 显示提示信息
        if fileName_choose != '':
            self.model_path = fileName_choose
            self.textEdit_model.setText(fileName_choose)
        else:
            self.textEdit_model.setText('Using defaut model')

        # 恢复界面
        gif = QMovie(':/images/ui/scan.gif')
        self.label_face.setMovie(gif)
        gif.start()

    def button_open_camera_click(self):
        if self.timer_camera.isActive() == False:  # 检查定时状态
            flag = self.cap.open(self.CAM_NUM)  # 检查相机状态
            if flag == False:  # 相机打开失败提示
                msg = QtWidgets.QMessageBox.warning(self, u"Warning",
                                                    u"请检测相机与电脑是否连接正确！ ",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)

            else:
                # 准备运行识别程序
                self.textEdit_pic.setText('No photo is choosed')
                QtWidgets.QApplication.processEvents()
                self.textEdit_camera.setText('Camera on...')
                self.label_face.setText('Analysing...\n\nleading')
                # 新建对象
                self.emotion_model = Emotion_Rec(self.model_path)
                QtWidgets.QApplication.processEvents()
                # 打开定时器
                self.timer_camera.start(30)
        else:
            # 定时器未开启，界面回复初始状态
            self.timer_camera.stop()
            self.cap.release()
            self.textEdit_camera.setText('Camera off')
            self.textEdit_pic.setText('No photo is choosed')
            self.label_face.clear()
            gif = QMovie(':/images/ui/scan.gif')
            self.label_face.setMovie(gif)
            gif.start()
            self.showResult()  # 条形图无数据

    def show_camera(self):
        # 定时器槽函数，每隔一段时间执行
        flag, self.image = self.cap.read()  # 获取画面
        self.image = cv2.flip(self.image, 1)  # 左右翻转

        time_start = time.time()  # 计时
        # 使用模型预测
        results, result = self.emotion_model.run(self.image, self.label_face)
        time_end = time.time()
        # 在界面显示结果
        self.showResult(results, result, time_end - time_start)

    def choose_pic(self):
        # 界面处理
        self.timer_camera.stop()
        self.cap.release()
        self.label_face.clear()
        self.textBrowser_result.setText('None')
        self.textBrowser_time.setText('0 s')
        self.textEdit_camera.setText('Camera off')
        self.showResult()

        # 使用文件选择对话框选择图片
        fileName_choose, filetype = QFileDialog.getOpenFileName(
            self, "Choose a photo...",
            self.path,  # 起始路径
            "图片(*.jpg;*.jpeg;*.png)")  # 文件类型
        self.path = fileName_choose  # 保存路径
        if fileName_choose != '':
            self.textEdit_pic.setText(fileName_choose)
            self.label_face.setText('Analysing...\n\nleading')
            QtWidgets.QApplication.processEvents()
            # 生成模型对象
            self.emotion_model = Emotion_Rec(self.model_path)

            # 计时并开始模型预测
            image = self.cv_imread(fileName_choose)  # 读取选择的图片
            QtWidgets.QApplication.processEvents()
            time_start = time.time()
            results, result = self.emotion_model.run(image, self.label_face)
            time_end = time.time()
            # 显示结果
            self.showResult(results, result, time_end - time_start)

        else:
            # 选择取消，恢复界面状态
            self.textEdit_pic.setText('no photo is choosed')
            gif = QMovie(':/images/ui/scan.gif')
            self.label_face.setMovie(gif)
            gif.start()
            self.showResult()

    def cv_imread(self, filePath):
        # 读取图片
        cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
        ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
        ## cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
        return cv_img

    def showResult(self, results=None, result='none', totalTime=0.0):
        if results is None:
            for emotion in self.EMOTIONS:
                bar_widget = self.findChild(QProgressBar, name='progressBar_' + emotion)
                bar_widget.setValue(0)
                text_widget = self.findChild(QTextBrowser, name='textBrowser_' + emotion)
                text_widget.setText(str(0) + '%')
        else:
            self.textBrowser_result.setText(result)
            self.textBrowser_time.setText(str(round(totalTime, 3)) + ' s')
            for (i, (emotion, prob)) in enumerate(results):
                # widget = QProgressBar(self)
                # widget.setObjectName(emotion)
                bar_widget = self.findChild(QProgressBar, name='progressBar_' + emotion)
                bar_widget.setValue(prob * 100)
                text_widget = self.findChild(QTextBrowser, name='textBrowser_' + emotion)
                prob = round(prob * 100, 2)
                text_widget.setText(str(prob) + '%')

