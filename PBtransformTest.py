import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import numpy as np
from tensorflow.python import debug as tfdbg
import os
import RGBSetBuilder


train_batch_count = 200
validation_batch_count = 100
test_batch_count = 100

data_path = 'E:/study/DL/HJFaceRecognition/project/model.pb'
pb_file_path = 'E:/study/DL/HJFaceRecognition/project/model1.pb'

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


trainBuilder = RGBSetBuilder.trainBuilder()
trainBuilder.decode_and_read()
print("trian")
print(trainBuilder.image_count)
print(trainBuilder.label_list)
print(trainBuilder.training_label_list)
print(trainBuilder.count_label)

validationBuilder = RGBSetBuilder.validationBuilder()
validationBuilder.decode_and_read()
print("validation")
print(validationBuilder.image_count)
print(validationBuilder.label_list)
print(validationBuilder.training_label_list)
print(validationBuilder.count_label)

testBuilder = RGBSetBuilder.testBuilder()
testBuilder.decode_and_read()
print("test")
print(testBuilder.image_count)
print(testBuilder.label_list)
print(testBuilder.training_label_list)
print(testBuilder.count_label)

trainHighBuilder = RGBSetBuilder.train_highBuilder()
trainHighBuilder.decode_and_read()
print("train high")
print(trainHighBuilder.image_count)
print(trainHighBuilder.label_list)
print(trainHighBuilder.training_label_list)
print(trainHighBuilder.count_label)

trainMiddleBuilder = RGBSetBuilder.train_middleBuilder()
trainMiddleBuilder.decode_and_read()
print("train middle")
print(trainMiddleBuilder.image_count)
print(trainMiddleBuilder.label_list)
print(trainMiddleBuilder.training_label_list)
print(trainMiddleBuilder.count_label)

trainLowBuilder = RGBSetBuilder.train_lowBuilder()
trainLowBuilder.decode_and_read()
print("train low")
print(trainLowBuilder.image_count)
print(trainLowBuilder.label_list)
print(trainLowBuilder.training_label_list)
print(trainLowBuilder.count_label)
os.system("pause")

mainSess = tf.InteractiveSession()
mainSess = tfdbg.LocalCLIDebugWrapperSession(mainSess)

'''
输入，占位符
'''
x = tf.placeholder("float",shape=[None,64*64,3],name='input666')
y_ = tf.placeholder("float",shape=[None,len(trainBuilder.label_list)])#占位符，其具体值由某一次具体计算决定

'''
归一化
'''
#x_image = tf.reshape(x, [1,64,64,3])
x_image = tf.reshape(x, [-1, 64, 64, 3])
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
y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2,name='output')


'''
类别预测和损失函数
'''
cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-8,1.0)))#损失函数为交叉熵

'''
训练模型&模型评估
'''
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
mainSess.run(tf.global_variables_initializer())



for i in range(2000):
    batch_train_xs,batch_train_ys,train_randomList = trainBuilder.fair_next_batch_image(batch_count=train_batch_count)
    batch_validation_xs,batch_validation_ys,validation_randomList = validationBuilder.fair_next_batch_image(batch_count=validation_batch_count)
    batch_train_high_xs,batch_train_high_ys,train_high_randomList = trainHighBuilder.fair_next_batch_image(batch_count=validation_batch_count)
    batch_train_middle_xs,batch_train_middle_ys,train_middle_randomList = trainMiddleBuilder.fair_next_batch_image(batch_count=30)
    batch_train_low_xs,batch_train_low_ys,train_low_randomList = trainLowBuilder.fair_next_batch_image(batch_count=30)
    if i%5 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch_train_xs, y_:batch_train_ys})
        print("step %d, 训练集准确率 %g"%(i, train_accuracy))
        train_cross_entropy = cross_entropy.eval(feed_dict={x:batch_train_xs, y_:batch_train_ys})
        train_cross_entropy = train_cross_entropy / train_batch_count
        print("step %d, 训练集交叉熵 %g"%(i, train_cross_entropy))

        validation_accuracy = accuracy.eval(feed_dict={x:batch_validation_xs, y_:batch_validation_ys})
        print("step %d, 验证集准确率 %g"%(i, validation_accuracy))
        validation_cross_entropy = cross_entropy.eval(feed_dict={x:batch_validation_xs, y_:batch_validation_ys})
        validation_cross_entropy = validation_cross_entropy / validation_batch_count
        print("step %d, 验证集交叉熵 %g"%(i, validation_cross_entropy))

        train_high_accuracy = accuracy.eval(feed_dict={x:batch_train_high_xs, y_:batch_train_high_ys})
        print("step %d, 验证集高频部分准确率 %g"%(i, train_high_accuracy))
        train_high_cross_entropy = cross_entropy.eval(feed_dict={x:batch_train_high_xs, y_:batch_train_high_ys})
        train_high_cross_entropy = train_high_cross_entropy / validation_batch_count
        print("step %d, 验证集高频部分交叉熵 %g"%(i, train_high_cross_entropy))

        train_middle_accuracy = accuracy.eval(feed_dict={x:batch_train_middle_xs, y_:batch_train_middle_ys})
        print("step %d, 验证集中频部分准确率 %g"%(i, train_middle_accuracy))
        train_middle_cross_entropy = cross_entropy.eval(feed_dict={x:batch_train_middle_xs, y_:batch_train_middle_ys})
        train_middle_cross_entropy = train_middle_cross_entropy / 30
        print("step %d, 验证集中频部分交叉熵 %g"%(i, train_middle_cross_entropy))

        train_low_accuracy = accuracy.eval(feed_dict={x:batch_train_low_xs, y_:batch_train_low_ys})
        print("step %d, 验证集低频部分准确率 %g"%(i, train_low_accuracy))
        train_low_cross_entropy = cross_entropy.eval(feed_dict={x:batch_train_low_xs, y_:batch_train_low_ys})
        train_low_cross_entropy = train_low_cross_entropy / 30
        print("step %d, 验证集低频部分交叉熵 %g"%(i, train_low_cross_entropy))
        print("---------------------------------")

        print(validation_randomList)
        print("__________________________________")
