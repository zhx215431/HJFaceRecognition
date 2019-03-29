import imageProcessor
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class RGBIP(imageProcessor.IP):
    def Recordfilename(self):
        filename = "RGBImage.tfrecords"
        return filename
#将原始的训练集编码生成二进制文件
    def create_record(self):
        writer = tf.python_io.TFRecordWriter(self.filename)
        for index, name in enumerate(self.classes):
            class_path = self.orig_picture + "/" + name + "/"
            for img_name in os.listdir(class_path):
                img_path = class_path + img_name
                img = Image.open(img_path)
                img = img.resize((self.imageLength,self.imageLength))#转换图片大小
                img_raw = img.tobytes()#转化为原生bytes
                print(index, "\n", img_raw)
                example = tf.train.Example(
                    features = tf.train.Features(feature={
                        'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [index])),
                        'img_raw': tf.train.Feature(bytes_list = tf.train.BytesList(value = [img_raw]))
                    }))
                writer.write(example.SerializeToString())
        writer.close()

#解码
    def read_and_decode(self):
        filename_queue = tf.train.string_input_producer([self.filename])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'label': tf.FixedLenFeature([], tf.int64),
                'img_raw': tf.FixedLenFeature([], tf.string)
            })
        label = features['label']
        img = features['img_raw']
        img = tf.decode_raw(img, tf.uint8)
        img = tf.reshape(img, [self.imageLength, self.imageLength, 3])
        label = tf.cast(label, tf.int32)
        return img, label

#将张量转换为灰度图像
    def show_gary_tensor_to_image(self, tensor):
        plt.imshow(tensor[:,:,0], cmap = 'gray')
        plt.show()
