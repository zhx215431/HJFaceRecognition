import RGBImageProcessor
import grayImageProcessor
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np

orig_picture = 'E:/study/DL/1(test)/'



def single_img_data(image_name):
    image_path = orig_picture + image_name
    img = Image.open(image_path)
    img = img.resize((64,64))

    img_np = np.array(img)
    #img_min = np.min(img_np)
    #img_max = np.max(img_np)
    #img_np = (img_np - img_min)/(img_max - img_min)

    #img_avg = np.average(img_np)
    #img_std = np.std(img_np)
    #print(img_min)
    #print(img_max)
    #print(img_avg)
    #print(img_std)
    #img_np = (img_np - img_avg)/img_std
    #print(img_np)

    return img_np



def tensorflow_nor(img_name):

    sess = tf.InteractiveSession()

    image_path = orig_picture + img_name
    img = Image.open(image_path)
    img_np = np.array(img)
    image_new = tf.image.per_image_standardization(img_np)



    plt.imshow(Image.fromarray(sess.run(image_new).astype('uint8')).convert('RGB'))
    #plt.imshow(Image.fromarray(img_np.astype('uint8')).convert('RGB'))
    plt.show()

    print(sess.run(image_new))
