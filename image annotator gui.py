from itsdangerous import json
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision as tv

from models import utils, caption
from datasets import coco
from configuration import Config
from engine import train_one_epoch, evaluate

from PIL import Image
import numpy as np
import random
import os
import sys
import json

from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow,QLabel,QStatusBar,QGridLayout,QVBoxLayout,QMenuBar,QFrame,QPushButton
from transformers import AutoModelForPreTraining, AutoTokenizer
from normalizer import normalize

from transformers import BertTokenizer
from datasets.utils import nested_tensor_from_tensor_list, read_json

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Window(QMainWindow):

    def __init__(self,image_names,captions,image_no = 0,caption_no=0, *args, **kwargs):
        QMainWindow.__init__(self,*args, **kwargs)
        self.image_no = image_no
        self.caption_no = caption_no
        self.image_names = image_names
        self.annotations = captions
        self.annotation_file_name = "new_annotations.json"
        self.setupUI()
        self.show()

    def set_photo_caption(self):

        filename = os.path.join(location,'images','images',self.image_names[self.image_no])
        filename_text = "File Name :" + self.image_names[self.image_no] 
            
        self.label.setText(filename_text)
        self.caption_label.setText("Caption of: "+self.image_names[self.caption_no])


        # prev_caption = '\n\n'
        # if self.caption_no != 0 :
        #     annotation_texts = self.annotations[self.caption_no- 1]
        #     prev_caption     = self.text_from_caption(annotation_texts) + '\n'
        #     #self.button1.setText(prev_caption)
        
        annotation_texts = self.annotations[self.caption_no]
        this_caption     = self.text_from_caption(annotation_texts) + '\n'
        self.button2.setText(this_caption)

        # annotation_texts = self.annotations[self.caption_no + 1]
        # next_caption     = self.text_from_caption(annotation_texts) + '\n'
        # #self.button3.setText(next_caption)

        self.photo.setPixmap(QtGui.QPixmap(filename))
        
        self.show()
    
    def text_from_caption(self,annotation_text):
        text = ''
        for sentence in annotation_text['caption']:
            text = text + sentence + '\n'
        
        return text
    
    def button_pressed(self,number):
        
        
        if number != 3:
    
            filename = self.image_names[self.image_no]
            caption_texts  = [texts for texts in self.annotations[self.caption_no]['caption']]

            if not os.path.exists(self.annotation_file_name):
                with open(self.annotation_file_name,'w',  encoding="utf8") as file:
                    pass
                print("created {}".format(self.annotation_file_name))

            with open(self.annotation_file_name,'r',  encoding="utf8") as file:
                try:
                    annotations = json.load(file)
                    annotations.append({'filename':filename,'caption':caption_texts})       
                except:
                    annotations = [{'filename':filename,'caption':caption_texts}]

            with open(self.annotation_file_name,'w',  encoding="utf8") as file:
                json.dump(annotations,file, ensure_ascii=False)
            
            self.caption_no += 1

        self.image_no += 1
        self.set_photo_caption()

    
    def dec_img(self):
        if self.image_no!=0:
            self.image_no -= 1
        if self.caption_no != 0:
            self.caption_no -= 1
        self.set_photo_caption()
    
    def change_caption(self,change):

        if self.caption_no > 0 or self.caption_no < len(self.annotations) - 1:

            self.caption_no += change
            self.set_photo_caption()


    def setupUI(self):
        self.central_widget = QWidget(self)
        #self.setStyleSheet('background-color:#222133')
        self.setObjectName("MainWindow")
        self.resize(800, 650)

        self.centralwidget = QWidget(self)
        self.centralwidget.setObjectName("centralwidget")

        self.button_prev = QPushButton(self.centralwidget)
        self.button_prev.setGeometry(QtCore.QRect(100, 10, 100, 50))
        self.button_prev.setText("Go Back")
        self.button_prev.clicked.connect(self.dec_img)

        self.photo = QLabel(self.centralwidget)
        self.photo.setGeometry(QtCore.QRect(200, 80, 450, 280))
        self.photo.setText("")
        self.photo.setScaledContents(True)
        self.photo.setObjectName("photo")

        
        self.label   = QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(350, 10, 200, 50))
        self.label.setFont(QFont('Arial', 15))



        self.inc_caption = QPushButton(self.centralwidget)
        self.inc_caption.setGeometry(QtCore.QRect(300, 380, 50, 50))
        self.inc_caption.setObjectName("Caption Offset Dec")
        self.inc_caption.setText('-')
        self.inc_caption.clicked.connect(lambda: self.change_caption(-1))

        self.caption_label   = QLabel(self.centralwidget)
        self.caption_label.setGeometry(QtCore.QRect(350, 380, 200, 50))
        self.caption_label.setFont(QFont('Arial', 15))

        self.dec_caption = QPushButton(self.centralwidget)
        self.dec_caption.setGeometry(QtCore.QRect(550, 380, 50, 50))
        self.dec_caption.setObjectName("Caption Offset Inc")
        self.dec_caption.setText('+')
        self.dec_caption.clicked.connect(lambda: self.change_caption(1))

        # self.button1 = QPushButton(self.centralwidget)
        # self.button1.setGeometry(QtCore.QRect(50, 270, 700, 80))
        # self.button1.setObjectName("Previous")
        
        self.button2 = QPushButton(self.centralwidget)
        self.button2.setGeometry(QtCore.QRect(50, 470, 700, 80))
        self.button2.setObjectName("Current")
        
        # self.button3 = QPushButton(self.centralwidget)
        # self.button3.setGeometry(QtCore.QRect(50, 470, 700, 80))
        # self.button3.setObjectName("Next")
        

        self.button4 = QPushButton(self.centralwidget)
        self.button4.setGeometry(QtCore.QRect(350, 580, 100, 50))
        self.button4.setObjectName("Cancel")
        self.button4.setText("Cancel")
        

        self.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(self)

        #self.button1.clicked.connect(lambda:self.button_pressed(0))
        self.button2.clicked.connect(lambda:self.button_pressed(1))
        #self.button3.clicked.connect(lambda:self.button_pressed(2))
        self.button4.clicked.connect(lambda:self.button_pressed(3))




if __name__ == '__main__':
    
    annotation_file_name = 'new_annotations.json'

    with open('captions.json','r',encoding="utf8") as file:
        captions = json.load(file)
    with open(annotation_file_name,'r',encoding="utf8") as file:
        annotations = json.load(file)
        image_no = annotations[-1]['filename'].split('.')[0]
        image_no = int(image_no) - 1
        caption_no = image_no 

    location = 'G:\\Pioneer Alpha\\Task 3'
    image_names = sorted([int(name.split('.')[0]) for name in os.listdir(os.path.join(location,'images','images'))])
    image_names = [str(name)+'.png' for name in image_names]

    app = QApplication(sys.argv)
    Gui = Window(image_names,captions,image_no,caption_no)
    Gui.show()
    Gui.set_photo_caption()
    sys.exit(app.exec_())
    

        
            
        


