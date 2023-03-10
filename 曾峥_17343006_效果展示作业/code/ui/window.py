# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'window.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(761, 606)
        Form.setStyleSheet("background-image: url(:/images/ui/background.png);\n"
"")
        self.textBrowser_3 = QtWidgets.QTextBrowser(Form)
        self.textBrowser_3.setGeometry(QtCore.QRect(140, 20, 531, 91))
        self.textBrowser_3.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser_3.setObjectName("textBrowser_3")
        self.textEdit_model = QtWidgets.QTextEdit(Form)
        self.textEdit_model.setGeometry(QtCore.QRect(70, 130, 381, 41))
        self.textEdit_model.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.textEdit_model.setStyleSheet("font: 75 15pt \"SimSun\";\n"
"color: rgb(255, 255, 255);\n"
"font-style:normal;")
        self.textEdit_model.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.textEdit_model.setReadOnly(True)
        self.textEdit_model.setObjectName("textEdit_model")
        self.textEdit_pic = QtWidgets.QTextEdit(Form)
        self.textEdit_pic.setGeometry(QtCore.QRect(70, 250, 381, 41))
        self.textEdit_pic.setStyleSheet("font: 75 16pt \"SimSun\";\n"
"color: rgb(255, 255, 255);\n"
"font-style:normal;")
        self.textEdit_pic.setReadOnly(True)
        self.textEdit_pic.setObjectName("textEdit_pic")
        self.textEdit_camera = QtWidgets.QTextEdit(Form)
        self.textEdit_camera.setGeometry(QtCore.QRect(70, 190, 381, 41))
        self.textEdit_camera.setStyleSheet("font: 75 16pt \"SimSun\";\n"
"color: rgb(255, 255, 255);\n"
"font-style:normal;")
        self.textEdit_camera.setReadOnly(True)
        self.textEdit_camera.setObjectName("textEdit_camera")
        self.label_5 = QtWidgets.QLabel(Form)
        self.label_5.setGeometry(QtCore.QRect(510, 170, 41, 41))
        self.label_5.setStyleSheet("border-image: url(:/images/ui/speed.png);")
        self.label_5.setText("")
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(Form)
        self.label_6.setGeometry(QtCore.QRect(510, 230, 41, 41))
        self.label_6.setStyleSheet("border-image: url(:/images/ui/result.png);")
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        self.textBrowser_7 = QtWidgets.QTextBrowser(Form)
        self.textBrowser_7.setGeometry(QtCore.QRect(560, 180, 201, 41))
        self.textBrowser_7.setStyleSheet("")
        self.textBrowser_7.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser_7.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textBrowser_7.setObjectName("textBrowser_7")
        self.textBrowser_8 = QtWidgets.QTextBrowser(Form)
        self.textBrowser_8.setGeometry(QtCore.QRect(560, 240, 191, 41))
        self.textBrowser_8.setStyleSheet("")
        self.textBrowser_8.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser_8.setObjectName("textBrowser_8")
        self.toolButton_model = QtWidgets.QToolButton(Form)
        self.toolButton_model.setGeometry(QtCore.QRect(20, 130, 41, 41))
        self.toolButton_model.setStyleSheet("border-image: url(:/images/ui/model.png);")
        self.toolButton_model.setText("")
        self.toolButton_model.setObjectName("toolButton_model")
        self.toolButton_camera = QtWidgets.QToolButton(Form)
        self.toolButton_camera.setGeometry(QtCore.QRect(20, 190, 41, 41))
        self.toolButton_camera.setStyleSheet("border-image: url(:/images/ui/camera.png);")
        self.toolButton_camera.setText("")
        self.toolButton_camera.setObjectName("toolButton_camera")
        self.toolButton_file = QtWidgets.QToolButton(Form)
        self.toolButton_file.setGeometry(QtCore.QRect(20, 250, 41, 41))
        self.toolButton_file.setStyleSheet("border-image: url(:/images/ui/file.png);")
        self.toolButton_file.setText("")
        self.toolButton_file.setObjectName("toolButton_file")
        self.textBrowser_time = QtWidgets.QTextBrowser(Form)
        self.textBrowser_time.setGeometry(QtCore.QRect(620, 180, 141, 41))
        self.textBrowser_time.setStyleSheet("font: 75 16pt \"SimSun\";\n"
"color: rgb(255, 255, 255);\n"
"font-style:normal;")
        self.textBrowser_time.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser_time.setObjectName("textBrowser_time")
        self.textBrowser_result = QtWidgets.QTextBrowser(Form)
        self.textBrowser_result.setGeometry(QtCore.QRect(640, 240, 101, 41))
        self.textBrowser_result.setStyleSheet("font: 75 16pt \"SimSun\";\n"
"color: rgb(255, 255, 255);\n"
"font-style:normal;")
        self.textBrowser_result.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser_result.setObjectName("textBrowser_result")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(Form)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(460, 310, 151, 281))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_emotions = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_emotions.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_emotions.setObjectName("verticalLayout_emotions")
        self.textBrowser_9 = QtWidgets.QTextBrowser(self.verticalLayoutWidget_2)
        self.textBrowser_9.setStyleSheet("")
        self.textBrowser_9.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser_9.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textBrowser_9.setObjectName("textBrowser_9")
        self.verticalLayout_emotions.addWidget(self.textBrowser_9)
        self.textBrowser_10 = QtWidgets.QTextBrowser(self.verticalLayoutWidget_2)
        self.textBrowser_10.setStyleSheet("")
        self.textBrowser_10.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser_10.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textBrowser_10.setObjectName("textBrowser_10")
        self.verticalLayout_emotions.addWidget(self.textBrowser_10)
        self.textBrowser_11 = QtWidgets.QTextBrowser(self.verticalLayoutWidget_2)
        self.textBrowser_11.setStyleSheet("")
        self.textBrowser_11.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser_11.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textBrowser_11.setObjectName("textBrowser_11")
        self.verticalLayout_emotions.addWidget(self.textBrowser_11)
        self.textBrowser_12 = QtWidgets.QTextBrowser(self.verticalLayoutWidget_2)
        self.textBrowser_12.setStyleSheet("")
        self.textBrowser_12.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser_12.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textBrowser_12.setObjectName("textBrowser_12")
        self.verticalLayout_emotions.addWidget(self.textBrowser_12)
        self.textBrowser_13 = QtWidgets.QTextBrowser(self.verticalLayoutWidget_2)
        self.textBrowser_13.setStyleSheet("")
        self.textBrowser_13.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser_13.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textBrowser_13.setObjectName("textBrowser_13")
        self.verticalLayout_emotions.addWidget(self.textBrowser_13)
        self.textBrowser_14 = QtWidgets.QTextBrowser(self.verticalLayoutWidget_2)
        self.textBrowser_14.setStyleSheet("")
        self.textBrowser_14.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser_14.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textBrowser_14.setObjectName("textBrowser_14")
        self.verticalLayout_emotions.addWidget(self.textBrowser_14)
        self.textBrowser_15 = QtWidgets.QTextBrowser(self.verticalLayoutWidget_2)
        self.textBrowser_15.setStyleSheet("")
        self.textBrowser_15.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser_15.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textBrowser_15.setObjectName("textBrowser_15")
        self.verticalLayout_emotions.addWidget(self.textBrowser_15)
        self.textBrowser_10.raise_()
        self.textBrowser_14.raise_()
        self.textBrowser_15.raise_()
        self.textBrowser_12.raise_()
        self.textBrowser_11.raise_()
        self.textBrowser_9.raise_()
        self.textBrowser_13.raise_()
        self.label_face = QtWidgets.QLabel(Form)
        self.label_face.setGeometry(QtCore.QRect(10, 300, 439, 299))
        self.label_face.setStyleSheet("border-image: url(:/images/ui/scan.gif);\n"
"font: 75 22pt \"SimSun\";\n"
"color: rgb(0, 0, 0);\n"
"\n"
"")
        self.label_face.setText("")
        self.label_face.setAlignment(QtCore.Qt.AlignCenter)
        self.label_face.setObjectName("label_face")
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(Form)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(560, 300, 160, 301))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout_progress = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_progress.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_progress.setObjectName("verticalLayout_progress")
        self.progressBar_angry = QtWidgets.QProgressBar(self.verticalLayoutWidget_3)
        self.progressBar_angry.setProperty("value", 24)
        self.progressBar_angry.setObjectName("progressBar_angry")
        self.verticalLayout_progress.addWidget(self.progressBar_angry)
        self.progressBar_disgust = QtWidgets.QProgressBar(self.verticalLayoutWidget_3)
        self.progressBar_disgust.setProperty("value", 24)
        self.progressBar_disgust.setObjectName("progressBar_disgust")
        self.verticalLayout_progress.addWidget(self.progressBar_disgust)
        self.progressBar_scared = QtWidgets.QProgressBar(self.verticalLayoutWidget_3)
        self.progressBar_scared.setProperty("value", 24)
        self.progressBar_scared.setObjectName("progressBar_scared")
        self.verticalLayout_progress.addWidget(self.progressBar_scared)
        self.progressBar_happy = QtWidgets.QProgressBar(self.verticalLayoutWidget_3)
        self.progressBar_happy.setProperty("value", 24)
        self.progressBar_happy.setObjectName("progressBar_happy")
        self.verticalLayout_progress.addWidget(self.progressBar_happy)
        self.progressBar_sad = QtWidgets.QProgressBar(self.verticalLayoutWidget_3)
        self.progressBar_sad.setProperty("value", 24)
        self.progressBar_sad.setObjectName("progressBar_sad")
        self.verticalLayout_progress.addWidget(self.progressBar_sad)
        self.progressBar_surprised = QtWidgets.QProgressBar(self.verticalLayoutWidget_3)
        self.progressBar_surprised.setProperty("value", 24)
        self.progressBar_surprised.setObjectName("progressBar_surprised")
        self.verticalLayout_progress.addWidget(self.progressBar_surprised)
        self.progressBar_neutral = QtWidgets.QProgressBar(self.verticalLayoutWidget_3)
        self.progressBar_neutral.setProperty("value", 24)
        self.progressBar_neutral.setObjectName("progressBar_neutral")
        self.verticalLayout_progress.addWidget(self.progressBar_neutral)
        self.verticalLayoutWidget = QtWidgets.QWidget(Form)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(690, 310, 160, 281))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout_nums = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_nums.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_nums.setObjectName("verticalLayout_nums")
        self.textBrowser_angry = QtWidgets.QTextBrowser(self.verticalLayoutWidget)
        self.textBrowser_angry.setStyleSheet("font: 75 14pt \"SimSun\";\n"
"color: rgb(255, 255, 255);\n"
"font-style:normal;")
        self.textBrowser_angry.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser_angry.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textBrowser_angry.setObjectName("textBrowser_angry")
        self.verticalLayout_nums.addWidget(self.textBrowser_angry)
        self.textBrowser_disgust = QtWidgets.QTextBrowser(self.verticalLayoutWidget)
        self.textBrowser_disgust.setStyleSheet("font: 75 14pt \"SimSun\";\n"
"color: rgb(255, 255, 255);\n"
"font-style:normal;")
        self.textBrowser_disgust.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser_disgust.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textBrowser_disgust.setObjectName("textBrowser_disgust")
        self.verticalLayout_nums.addWidget(self.textBrowser_disgust)
        self.textBrowser_scared = QtWidgets.QTextBrowser(self.verticalLayoutWidget)
        self.textBrowser_scared.setStyleSheet("font: 75 14pt \"SimSun\";\n"
"color: rgb(255, 255, 255);\n"
"font-style:normal;")
        self.textBrowser_scared.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser_scared.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textBrowser_scared.setObjectName("textBrowser_scared")
        self.verticalLayout_nums.addWidget(self.textBrowser_scared)
        self.textBrowser_happy = QtWidgets.QTextBrowser(self.verticalLayoutWidget)
        self.textBrowser_happy.setStyleSheet("font: 75 14pt \"SimSun\";\n"
"color: rgb(255, 255, 255);\n"
"font-style:normal;")
        self.textBrowser_happy.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser_happy.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textBrowser_happy.setObjectName("textBrowser_happy")
        self.verticalLayout_nums.addWidget(self.textBrowser_happy)
        self.textBrowser_sad = QtWidgets.QTextBrowser(self.verticalLayoutWidget)
        self.textBrowser_sad.setStyleSheet("font: 75 14pt \"SimSun\";\n"
"color: rgb(255, 255, 255);\n"
"font-style:normal;")
        self.textBrowser_sad.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser_sad.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textBrowser_sad.setObjectName("textBrowser_sad")
        self.verticalLayout_nums.addWidget(self.textBrowser_sad)
        self.textBrowser_surprised = QtWidgets.QTextBrowser(self.verticalLayoutWidget)
        self.textBrowser_surprised.setStyleSheet("font: 75 14pt \"SimSun\";\n"
"color: rgb(255, 255, 255);\n"
"font-style:normal;")
        self.textBrowser_surprised.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser_surprised.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textBrowser_surprised.setObjectName("textBrowser_surprised")
        self.verticalLayout_nums.addWidget(self.textBrowser_surprised)
        self.textBrowser_neutral = QtWidgets.QTextBrowser(self.verticalLayoutWidget)
        self.textBrowser_neutral.setStyleSheet("font: 75 14pt \"SimSun\";\n"
"color: rgb(255, 255, 255);\n"
"font-style:normal;")
        self.textBrowser_neutral.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser_neutral.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textBrowser_neutral.setObjectName("textBrowser_neutral")
        self.verticalLayout_nums.addWidget(self.textBrowser_neutral)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.textBrowser_3.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:26pt; font-weight:600; color:#ffffff;\">Emotion Recognition System</span></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600; font-style:italic; color:#ffffff;\">                </span></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600; font-style:italic; color:#ffffff;\">                     17343006 ??????</span></p></body></html>"))
        self.textEdit_model.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:15pt; font-weight:72; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:16pt; font-weight:400; color:#ffffff;\">Using defaut model</span></p></body></html>"))
        self.textEdit_pic.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:16pt; font-weight:72; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt; font-weight:400; color:#ffffff;\">Choose a photo</span></p></body></html>"))
        self.textEdit_camera.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:16pt; font-weight:72; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt; font-weight:400; color:#ffffff;\">Camera off</span></p></body></html>"))
        self.textBrowser_7.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:16pt; color:#ffffff;\">Time???</span></p></body></html>"))
        self.textBrowser_8.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:16pt; color:#ffffff;\">Result???</span></p></body></html>"))
        self.textBrowser_time.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:16pt; font-weight:72; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:400; color:#ffffff;\">0 s</span></p></body></html>"))
        self.textBrowser_result.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:16pt; font-weight:72; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:400; color:#ffffff;\">none</span></p></body></html>"))
        self.textBrowser_9.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt; color:#ffffff;\">angry</span></p></body></html>"))
        self.textBrowser_10.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt; color:#ffffff;\">disgust</span></p></body></html>"))
        self.textBrowser_11.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt; color:#ffffff;\">scared</span></p></body></html>"))
        self.textBrowser_12.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt; color:#ffffff;\">happy</span></p></body></html>"))
        self.textBrowser_13.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt; color:#ffffff;\">sad</span></p></body></html>"))
        self.textBrowser_14.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt; color:#ffffff;\">surprised</span></p></body></html>"))
        self.textBrowser_15.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt; color:#ffffff;\">neutral</span></p></body></html>"))
        self.textBrowser_angry.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:14pt; font-weight:72; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:400; color:#ffffff;\">90.89%</span></p></body></html>"))
        self.textBrowser_disgust.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:14pt; font-weight:72; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:400; color:#ffffff;\">0%</span></p></body></html>"))
        self.textBrowser_scared.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:14pt; font-weight:72; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:400; color:#ffffff;\">0%</span></p></body></html>"))
        self.textBrowser_happy.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:14pt; font-weight:72; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:400; color:#ffffff;\">0%</span></p></body></html>"))
        self.textBrowser_sad.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:14pt; font-weight:72; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:400; color:#ffffff;\">0%</span></p></body></html>"))
        self.textBrowser_surprised.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:14pt; font-weight:72; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:400; color:#ffffff;\">0%</span></p></body></html>"))
        self.textBrowser_neutral.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:14pt; font-weight:72; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:400; color:#ffffff;\">0%</span></p></body></html>"))
