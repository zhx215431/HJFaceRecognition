'''
问题阐述：
直接用训练用的计算图训练并生成的pb文件转换成的mlmodel文件具有以下问题
1.文件太大
2.存在一个不必要的占位符（用于验证以及做梯度下降的）
3.输入形状不符（由于tfrecord文件格式要求，图片要被保存为一个一维数组）

解决方案：
对已经生成的用于训练的PB文件，抓取其特定的卷积核以及偏置参数，生成一个新的计算图并且执行后生成PB文件
其转换成的mlmodel文件是最终可以放入app中的文件

该文件用于抓取训练用pb文件的参数并创建新的计算图，执行后生成新的pb文件
'''
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import numpy as np
from tensorflow.python import debug as tfdbg
import os

data_path = 'E:/study/DL/HJFaceRecognition/project/model.pb'
pb_file_path = 'E:/study/DL/HJFaceRecognition/project/model1.pb'


'''
显示计算图
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

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


mainSess = tf.InteractiveSession()
mainSess = tfdbg.LocalCLIDebugWrapperSession(mainSess)

'''
输入，占位符
'''
x = tf.placeholder("float",shape=[None,64,64,3],name='input666')

'''
归一化
'''
#x_image = tf.reshape(x, [1,64,64,3])
x_image = x
x_nor_image = x_image / 255

'''
第一层卷积
'''
W_conv1 = getParam(name='W_conv1:0')
b_conv1 = getParam(name='b_conv1:0')
h_conv1 = tf.nn.leaky_relu(conv2d(x_nor_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

'''
第二层卷积
'''
W_conv2 = getParam(name='W_conv2:0')
b_conv2 = getParam(name='b_conv2:0')
h_conv2 = tf.nn.leaky_relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

'''
密集连接层
'''
W_fc1 = getParam(name='W_fc1:0')
b_fc1 = getParam(name='b_fc1:0')
h_pool2_flat = tf.reshape(h_pool2,[-1, 16 * 16 * 40])
h_fc1 = tf.nn.leaky_relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

'''
输出层
'''
W_fc2 = getParam(name='W_fc2:0')
b_fc2 = getParam(name='b_fc2:0')
y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2, name='output')

mainSess.run(tf.global_variables_initializer())

constant_graph = graph_util.convert_variables_to_constants(mainSess, mainSess.graph_def, ['output'])
with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
        f.write(constant_graph.SerializeToString())
