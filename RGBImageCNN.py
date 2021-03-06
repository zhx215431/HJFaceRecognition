import tensorflow as tf
import RGBSetBuilder
from tensorflow.python import debug as tfdbg
import os
import Excel_Write
from tensorflow.python.framework import graph_util

excelFilePath = 'E:/study/DL/HJFaceRecognition/project/ExcelR&W/test.xls'
pb_file_path = 'E:/study/DL/HJFaceRecognition/project/model.pb'
train_batch_count = 50
validation_batch_count = 20
test_batch_count = 100

#权重初始化
def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.1, name='weight_init')
    return tf.Variable(initial,name=name)
#偏置初始化
def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape, name='bias_init')
    return tf.Variable(initial,name=name)
#卷积 1步长（stride size）,0边距(padding size)
#padding = 'SAME'要对边界进行0填充
#padding = 'VALID'不考虑
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#池化
#ksize 池化窗口的大小
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

#能够在运行图的时候，插入一些计算图
sess = tf.InteractiveSession()
# sess = tf.Session() 需要在启动session之前构建整个计算图，然后启动该计算图
sess = tfdbg.LocalCLIDebugWrapperSession(sess)

'''
占位符
'''
x = tf.placeholder("float",shape=[None,64*64,3],name='input')
y_ = tf.placeholder("float",shape=[None,len(trainBuilder.label_list)])#占位符，其具体值由某一次具体计算决定


'''
归一化，很重要
'''
x_image = tf.reshape(x, [-1, 64, 64, 3])
x_nor_image = x_image / 255

'''
第一层卷积
'''
#-1表示缺省
W_conv1 = weight_variable([4, 4, 3, 20],name='W_conv1')
b_conv1 = bias_variable([20],name='b_conv1')
h_conv1 = tf.nn.leaky_relu(conv2d(x_nor_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

'''
第二层卷积
'''
W_conv2 = weight_variable([3, 3, 20, 40],name='W_conv2')
b_conv2 = bias_variable([40],name='b_conv2')
h_conv2 = tf.nn.leaky_relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

'''
密集连接层
'''
W_fc1 = weight_variable([16 * 16 * 40, 1024],name='W_fc1')
b_fc1 = bias_variable([1024],name='b_fc1')
h_pool2_flat = tf.reshape(h_pool2,[-1, 16 * 16 * 40])
h_fc1 = tf.nn.leaky_relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

'''
DROUP OUT
'''
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

'''
输出层
'''
W_fc2 = weight_variable([1024, len(trainBuilder.label_list)],name='W_fc2')
b_fc2 = bias_variable([len(trainBuilder.label_list)],name='b_fc2')
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2,name='op')
'''
类别预测和损失函数
'''
cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-8,1.0)))#损失函数为交叉熵


'''
训练模型&模型评估
'''
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.global_variables_initializer())


write = Excel_Write.write(excelFilePath,4)


for i in range(100):
    batch_train_xs,batch_train_ys,train_randomList = trainBuilder.next_batch_image(batch_count=train_batch_count)
    batch_validation_xs,batch_validation_ys,validation_randomList = validationBuilder.fair_next_batch_image(batch_count=validation_batch_count)
    #batch_train_high_xs,batch_train_high_ys,train_high_randomList = trainHighBuilder.fair_next_batch_image(batch_count=validation_batch_count)
    #batch_train_middle_xs,batch_train_middle_ys,train_middle_randomList = trainMiddleBuilder.fair_next_batch_image(batch_count=30)
    #batch_train_low_xs,batch_train_low_ys,train_low_randomList = trainLowBuilder.fair_next_batch_image(batch_count=30)
    if i%5 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch_train_xs, y_:batch_train_ys, keep_prob:1.0})
        print("step %d, 训练集准确率 %g"%(i, train_accuracy))
        train_cross_entropy = cross_entropy.eval(feed_dict={x:batch_train_xs, y_:batch_train_ys, keep_prob:1.0})
        train_cross_entropy = train_cross_entropy / train_batch_count
        print("step %d, 训练集交叉熵 %g"%(i, train_cross_entropy))

        validation_accuracy = accuracy.eval(feed_dict={x:batch_validation_xs, y_:batch_validation_ys, keep_prob:1.0})
        print("step %d, 验证集准确率 %g"%(i, validation_accuracy))
        validation_cross_entropy = cross_entropy.eval(feed_dict={x:batch_validation_xs, y_:batch_validation_ys, keep_prob:1.0})
        validation_cross_entropy = validation_cross_entropy / validation_batch_count
        print("step %d, 验证集交叉熵 %g"%(i, validation_cross_entropy))

        #train_high_accuracy = accuracy.eval(feed_dict={x:batch_train_high_xs, y_:batch_train_high_ys, keep_prob:1.0})
        #print("step %d, 验证集高频部分准确率 %g"%(i, train_high_accuracy))
        #train_high_cross_entropy = cross_entropy.eval(feed_dict={x:batch_train_high_xs, y_:batch_train_high_ys, keep_prob:1.0})
        #train_high_cross_entropy = train_high_cross_entropy / validation_batch_count
        #print("step %d, 验证集高频部分交叉熵 %g"%(i, train_high_cross_entropy))

        #train_middle_accuracy = accuracy.eval(feed_dict={x:batch_train_middle_xs, y_:batch_train_middle_ys, keep_prob:1.0})
        #print("step %d, 验证集中频部分准确率 %g"%(i, train_middle_accuracy))
        #train_middle_cross_entropy = cross_entropy.eval(feed_dict={x:batch_train_middle_xs, y_:batch_train_middle_ys, keep_prob:1.0})
        #train_middle_cross_entropy = train_middle_cross_entropy / 30
        #print("step %d, 验证集中频部分交叉熵 %g"%(i, train_middle_cross_entropy))

        #train_low_accuracy = accuracy.eval(feed_dict={x:batch_train_low_xs, y_:batch_train_low_ys, keep_prob:1.0})
        #print("step %d, 验证集低频部分准确率 %g"%(i, train_low_accuracy))
        #train_low_cross_entropy = cross_entropy.eval(feed_dict={x:batch_train_low_xs, y_:batch_train_low_ys, keep_prob:1.0})
        #train_low_cross_entropy = train_low_cross_entropy / 30
        #print("step %d, 验证集低频部分交叉熵 %g"%(i, train_low_cross_entropy))
        #print("---------------------------------")
        #write.write_append(int(i/5 + 1),train_accuracy,train_cross_entropy,validation_accuracy,validation_cross_entropy)
        #write.write_exappend(int(i/5 + 1),train_high_accuracy,train_high_cross_entropy,train_middle_accuracy,train_middle_cross_entropy,train_low_accuracy,train_low_cross_entropy)
    train_step.run(feed_dict={x:batch_train_xs, y_:batch_train_ys, keep_prob: 0.5})
#test_xs, test_ys, test_randomList = testBuilder.fair_next_batch_image(batch_count=test_batch_count)
#print("测试集准确率 %g"%accuracy.eval(feed_dict={x:test_xs, y_:test_ys, keep_prob: 1.0}))
constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op'])
with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
        f.write(constant_graph.SerializeToString())
