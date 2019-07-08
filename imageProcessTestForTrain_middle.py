'''
生成训练集中频部分二进制文件
'''
import RGBImageProcessor
import grayImageProcessor
import tensorflow as tf
import PIL as Image
import matplotlib.pyplot as plt
import os

orig_picture = 'E:/study/DL/1.2(train_middle)'
gen_picture = 'E:/study/DL/1.2(train_middle)'
RecordFileName = "RGBImage_train_middle.tfrecords"

#t = grayImageProcessor.GaryIP(orig_picture, gen_picture)
t = RGBImageProcessor.RGBIP(orig_picture, gen_picture, RecordFileName)
print(t.classes)
print(t.num_samples)
os.system("pause")
t.create_record()
batch = t.read_and_decode()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    therads = tf.train.start_queue_runners(coord = coord)

    for i in range(t.num_samples):
        example, lab = sess.run(batch)
        shape = tf.shape(example)
        print(sess.run(shape))
        t.show_tensor_to_image(example)
        print(lab)
    coord.request_stop()
    coord.join(therads)
    sess.close()
