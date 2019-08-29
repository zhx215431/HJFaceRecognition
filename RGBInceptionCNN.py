import tensorflow as tf
import RGBSetBuilder
from tensorflow.python import debug as tfdbg
import os
import Excel_Write
from tensorflow.python.framework import graph_util

excelFilePath = 'E:/study/DL/HJFaceRecognition/project/ExcelR&W/test.xls'
pb_file_path = 'E:/study/DL/HJFaceRecognition/project/InceptionModel.pb'
train_batch_count = 50
validation_batch_count = 20
test_batch_count = 100

def weight_variable(shape,name):
    '''
    权重初始化
    正态分布
    标准差 0.1
    均值 0
    '''
    initial = tf.truncated_normal(shape, mean=0.0, stddev=0.1, name='weight_init')
    return tf.Variable(initial,name=name)

def bias_variable(shape,name):
    '''
    偏置初始化
    初始值为0.1
    '''
    initial = tf.constant(0.1, shape=shape, name='bias_init')
    return tf.Variable(initial,name=name)

def convWithStride(x,W,stride):
    '''
    卷积
    对边界进行0填充
    '''
    return tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding='SAME')


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

sess = tf.InteractiveSession()
sess = tfdbg.LocalCLIDebugWrapperSession(sess)


#占位符
x = tf.placeholder("float",shape=[None,64*64,3],name='input')
y_ = tf.placeholder("float",shape=[None,len(trainBuilder.label_list)],name='label')

#归一化
x_image = tf.reshape(x,[-1,64,64,3])
x_nor_image = x_image / 255
## TODO: 中心化


#stem
W_stem_conv1 = weight_variable([3,3,3,32],name='W_stem_conv1')
b_stem_conv1 = bias_variable([32],name='b_stem_conv1')
h_stem_conv1 = convWithStride(x_nor_image,W_stem_conv1,2)+ b_stem_conv1

W_stem_conv2 = weight_variable([3,3,32,64],name='W_stem_conv2')
b_stem_conv2 = bias_variable([64],name='b_stem_conv2')
h_stem_conv2 = tf.nn.relu(convWithStride(h_stem_conv1,W_stem_conv2,1) + b_stem_conv2,name='h_stem_conv2')

h_stem_conv3_part1 = tf.nn.max_pool(h_stem_conv2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

W_stem_conv3_part2 = weight_variable([3,3,64,96],name='W_stem_conv3_part2')
b_stem_conv3_part2 = bias_variable([96],name='h_stem_conv3_part2')
h_stem_conv3_part2 = convWithStride(h_stem_conv2,W_stem_conv3_part2,2) + b_stem_conv3_part2

h_stem_conv3 = tf.nn.relu(tf.concat([h_stem_conv3_part1,h_stem_conv3_part2],3),name='h_stem_conv3')

#inception
h_inception_avg_pooling = tf.nn.avg_pool(h_stem_conv3, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
W_inception_part1 = weight_variable([1,1,160,96],name='W_inception_part1')
b_inception_part1 = bias_variable([96],name='b_inception_part1')
h_inception_part1 = convWithStride(h_inception_avg_pooling,W_inception_part1,1) + b_inception_part1

W_inception_part2 = weight_variable([1,1,160,96],name='W_inception_part2')
b_inception_part2 = bias_variable([96],name='b_inception_part2')
h_inception_part2 = convWithStride(h_stem_conv3,W_inception_part2,1) + b_inception_part2

W_inception_part3_1 = weight_variable([1,1,160,64],name='W_inception_part3_1')
b_inception_part3_1 = bias_variable([64],name='b_inception_part3_1')
h_inception_part3_1 = convWithStride(h_stem_conv3,W_inception_part3_1,1) + b_inception_part3_1
W_inception_part3_2 = weight_variable([3,3,64,96],name='W_inception_part3_2')
b_inception_part3_2 = bias_variable([96],name='b_inception_part3_2')
h_inception_part3_2 = convWithStride(h_inception_part3_1,W_inception_part3_2,1) + b_inception_part3_2
h_inception_part3 = h_inception_part3_2

W_inception_part4_1 = weight_variable([1,1,160,64],name='W_inception_part4_1')
b_inception_part4_1 = bias_variable([64],name='b_inception_part4_1')
h_inception_part4_1 = convWithStride(h_stem_conv3,W_inception_part4_1,1) + b_inception_part4_1
W_inception_part4_2 = weight_variable([3,3,64,96],name='W_inception_part4_2')
b_inception_part4_2 = bias_variable([96],name='b_inception_part4_2')
h_inception_part4_2 = convWithStride(h_inception_part4_1,W_inception_part4_2,1) + b_inception_part4_2
W_inception_part4_3 = weight_variable([3,3,96,96],name='W_inception_part4_3')
b_inception_part4_3 = bias_variable([96],name='b_inception_part4_3')
h_inception_part4_3 = convWithStride(h_inception_part4_2,W_inception_part4_3,1) + b_inception_part4_3
h_inception_part4 = h_inception_part4_3

h_inception = tf.concat([h_inception_part1,h_inception_part2,h_inception_part3,h_inception_part4],3)
h_inception = tf.nn.relu(h_inception)

#reduction
h_reduction_part1 = tf.nn.max_pool(h_inception,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

W_reduction_part2_1 = weight_variable([1,1,384,192],name='W_reduction_part2_1')
b_reduction_part2_1 = bias_variable([192],name='b_reduction_part2_1')
h_reduction_part2_1 = convWithStride(h_inception,W_reduction_part2_1,1) + b_reduction_part2_1
W_reduction_part2_2 = weight_variable([3,3,192,192],name='W_reduction_part2_2')
b_reduction_part2_2 = bias_variable([192],name='b_reduction_part2_2')
h_reduction_part2 = convWithStride(h_reduction_part2_1,W_reduction_part2_2,2) + b_reduction_part2_2

h_reduction = tf.concat([h_reduction_part1,h_reduction_part2],3)
h_reduction = tf.nn.relu(h_reduction)

#avg pooling
h_avg_pool = tf.nn.avg_pool(h_reduction,ksize=[1,8,8,1],strides=[1,1,1,1],padding='VALID')
h_avg_pool = tf.reshape(h_avg_pool,[-1,576])

#dropout
keep_prob = tf.placeholder("float")
h_drop = tf.nn.dropout(h_avg_pool,keep_prob)

#softmax
W_softmax_FC = weight_variable([576,len(trainBuilder.label_list)],name='W_softmax_FC')
b_softmax_FC = bias_variable([len(trainBuilder.label_list)],name='b_softmax_FC')
y_conv = tf.nn.softmax(tf.matmul(h_drop,W_softmax_FC) + b_softmax_FC, name='op')

#类别预测，损失函数
cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-8,1.0)))


#训练模型&模型评估
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.global_variables_initializer())


write = Excel_Write.write(excelFilePath,5)


for i in range(5000):
    batch_train_xs,batch_train_ys,train_randomList = trainBuilder.next_batch_image(batch_count=train_batch_count)
    batch_validation_xs,batch_validation_ys,validation_randomList = validationBuilder.fair_next_batch_image(batch_count=validation_batch_count)
    #a = tf.shape(batch_train_xs)
    #b = tf.shape(batch_train_ys)
    #print(sess.run(a))
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
        write.write_append(int(i/5 + 1),train_accuracy,train_cross_entropy,validation_accuracy,validation_cross_entropy)
        #write.write_exappend(int(i/5 + 1),train_high_accuracy,train_high_cross_entropy,train_middle_accuracy,train_middle_cross_entropy,train_low_accuracy,train_low_cross_entropy)
    train_step.run(feed_dict={x:batch_train_xs, y_:batch_train_ys, keep_prob: 0.8})
#test_xs, test_ys, test_randomList = testBuilder.fair_next_batch_image(batch_count=test_batch_count)
#print("测试集准确率 %g"%accuracy.eval(feed_dict={x:test_xs, y_:test_ys, keep_prob: 1.0}))


constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op'])
with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
        f.write(constant_graph.SerializeToString())
