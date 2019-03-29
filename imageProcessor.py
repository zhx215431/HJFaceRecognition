import tensorflow as tf
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class IP:
    '''
    @param orig_picture = 读取训练集的初始路径
    @param gen_picture 生成二进制文件的路径
    @param classes = {'1','2',...} 标签集合
    @param num_samples = 120 样本数量
    @param filename 生成二进制文件的名称
    @param length 图片缩放后的长度
    '''
    def Recordfilename(self):
        filename = "子类重写"
        return filename

#参数设定
    def read_image(self):
        for parent, dirnames, filenames in os.walk(self.orig_picture):
            for dirname in dirnames:
                if dirname not in self.classes:
                    self.classes.append(dirname)
            for filename in filenames:
                self.num_samples = self.num_samples + 1

#初始化@param classes, @param num_samples
    def __init__(self, orig_picture, gen_picture):
        self.orig_picture = orig_picture
        self.gen_picture = gen_picture
        self.classes = []
        self.num_samples = 0
        self.imageLength = 64
        self.filename = self.Recordfilename()
        self.read_image()
#展示图片
    def show_image(self, path):
        image_raw_data = tf.gfile.GFile(path).read()
        with tf.Session() as sess:
            img_data = tf.image.decode_jpeg(image_raw_data)
            print(img_data.eval())
            plt.imshow(img_data.eval())
            plt.show()

#将张量转换为图片并展示
    def show_tensor_to_image(self, tensor):
        plt.imshow(tensor)
        plt.show()

#将原始的训练集编码生成二进制文件
    def create_record(self):
        print("子类重写")

#解码
    def read_and_decode(self):
        print("子类重写")
