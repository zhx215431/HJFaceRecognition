'''
描述见PBtransform.py
'''
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import numpy as np
from tensorflow.python import debug as tfdbg
import os
import RGBSetBuilder

import matplotlib.pyplot as plt
import imageProcessTestForTest_Singel

data_path = 'E:/study/DL/HJFaceRecognition/project/InceptionModel.pb'
pb_file_path = 'E:/study/DL/HJFaceRecognition/project/InceptionModel_useful.pb'


'''显示计算图
graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
graph_def.ParseFromString(tf.gfile.FastGFile(data_path,'rb').read())
tf.import_graph_def(graph_def, name='graph')
summaryWriter = tf.summary.FileWriter('log/',graph)
'''
def getParam(name):
    sess = tf.Session()
    with gfile.FastGFile(data_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        return sess.run(name)
    sess.close()

def convWithStride(x,W,stride):
    '''
    卷积
    对边界进行0填充
    '''
    return tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding='SAME')

mainSess = tf.InteractiveSession()
#mainSess = tfdbg.LocalCLIDebugWrapperSession(mainSess)

'''
输入 占位符
'''
x = tf.placeholder("float",shape=[None,64,64,3],name='input2333')

'''
归一化
'''
#x_image = tf.reshape(x,[-1,64,64,3]);
x_image = x
x_nor_image = x_image / 255

'''
stem
'''
W_stem_conv1 = getParam('W_stem_conv1:0')
b_stem_conv1 = getParam('b_stem_conv1:0')
h_stem_conv1 = convWithStride(x_nor_image,W_stem_conv1,2)+ b_stem_conv1

W_stem_conv2 = getParam('W_stem_conv2:0')
b_stem_conv2 = getParam('b_stem_conv2:0')
h_stem_conv2 = tf.nn.relu(convWithStride(h_stem_conv1,W_stem_conv2,1) + b_stem_conv2,name='h_stem_conv2')

h_stem_conv3_part1 = tf.nn.max_pool(h_stem_conv2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

W_stem_conv3_part2 = getParam('W_stem_conv3_part2:0')
b_stem_conv3_part2 = getParam('h_stem_conv3_part2:0')
h_stem_conv3_part2 = convWithStride(h_stem_conv2,W_stem_conv3_part2,2) + b_stem_conv3_part2

h_stem_conv3 = tf.nn.relu(tf.concat([h_stem_conv3_part1,h_stem_conv3_part2],3),name='h_stem_conv3')

'''
inception
'''
h_inception_avg_pooling = tf.nn.avg_pool(h_stem_conv3, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
W_inception_part1 = getParam('W_inception_part1:0')
b_inception_part1 = getParam('b_inception_part1:0')
h_inception_part1 = convWithStride(h_inception_avg_pooling,W_inception_part1,1) + b_inception_part1

W_inception_part2 = getParam('W_inception_part2:0')
b_inception_part2 = getParam('b_inception_part2:0')
h_inception_part2 = convWithStride(h_stem_conv3,W_inception_part2,1) + b_inception_part2

W_inception_part3_1 = getParam('W_inception_part3_1:0')
b_inception_part3_1 = getParam('b_inception_part3_1:0')
h_inception_part3_1 = convWithStride(h_stem_conv3,W_inception_part3_1,1) + b_inception_part3_1
W_inception_part3_2 = getParam('W_inception_part3_2:0')
b_inception_part3_2 = getParam('b_inception_part3_2:0')
h_inception_part3_2 = convWithStride(h_inception_part3_1,W_inception_part3_2,1) + b_inception_part3_2
h_inception_part3 = h_inception_part3_2

W_inception_part4_1 = getParam('W_inception_part4_1:0')
b_inception_part4_1 = getParam('b_inception_part4_1:0')
h_inception_part4_1 = convWithStride(h_stem_conv3,W_inception_part4_1,1) + b_inception_part4_1
W_inception_part4_2 = getParam('W_inception_part4_2:0')
b_inception_part4_2 = getParam('b_inception_part4_2:0')
h_inception_part4_2 = convWithStride(h_inception_part4_1,W_inception_part4_2,1) + b_inception_part4_2
W_inception_part4_3 = getParam('W_inception_part4_3:0')
b_inception_part4_3 = getParam('b_inception_part4_3:0')
h_inception_part4_3 = convWithStride(h_inception_part4_2,W_inception_part4_3,1) + b_inception_part4_3
h_inception_part4 = h_inception_part4_3

h_inception = tf.concat([h_inception_part1,h_inception_part2,h_inception_part3,h_inception_part4],3)
h_inception = tf.nn.relu(h_inception)


'''
reduction
'''
h_reduction_part1 = tf.nn.max_pool(h_inception,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

W_reduction_part2_1 = getParam('W_reduction_part2_1:0')
b_reduction_part2_1 = getParam('b_reduction_part2_1:0')
h_reduction_part2_1 = convWithStride(h_inception,W_reduction_part2_1,1) + b_reduction_part2_1
W_reduction_part2_2 = getParam('W_reduction_part2_2:0')
b_reduction_part2_2 = getParam('b_reduction_part2_2:0')
h_reduction_part2 = convWithStride(h_reduction_part2_1,W_reduction_part2_2,2) + b_reduction_part2_2

h_reduction = tf.concat([h_reduction_part1,h_reduction_part2],3)
h_reduction = tf.nn.relu(h_reduction)

'''
avg pooling
'''
h_avg_pool = tf.nn.avg_pool(h_reduction,ksize=[1,8,8,1],strides=[1,1,1,1],padding='VALID')
h_avg_pool = tf.reshape(h_avg_pool,[-1,576])


'''
softmax
'''
W_softmax_FC = getParam('W_softmax_FC:0')
b_softmax_FC = getParam('b_softmax_FC:0')
y_conv = tf.nn.softmax(tf.matmul(h_avg_pool,W_softmax_FC) + b_softmax_FC, name='output')


mainSess.run(tf.global_variables_initializer())

#validationBuilder = RGBSetBuilder.validationBuilder()
#validationBuilder.decode_and_read()
#batch_image_list,batch_label_list,random_class_list = validationBuilder.certain_batch_image()

#用于单张图片的测试
#singel_image = imageProcessTestForTest_Singel.single_img_data('IMG_4952.JPG')
#y = y_conv.eval(feed_dict={x:[singel_image]})
#print(y)

#pict = batch_image_list[0]
#pict = pict.reshape([64,64,3])
#plt.imshow(pict)
#plt.show()
#print(batch_label_list)

#生成新的模型
constant_graph = graph_util.convert_variables_to_constants(mainSess, mainSess.graph_def, ['output'])
with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
        f.write(constant_graph.SerializeToString())
