import setBuilder
import RGBImageProcessor
import tensorflow as tf
import PIL as Image
import matplotlib.pyplot as plt
import numpy as np

orig_test_picture = 'E:/study/DL/1(test)'
gen_test_picture = 'E:/study/DL/1(test)'
test_fileName = "RGBImage_test.tfrecords"

orig_train_picture = 'E:/study/DL/1(train)'
gen_train_picture = 'E:/study/DL/1(train)'
train_fileName = "RGBImage_train.tfrecords"

orig_validation_picture = 'E:/study/DL/1(validation)'
gen_validation_picture = 'E:/study/DL/1(validation)'
validation_fileName = "RGBImage_validation.tfrecords"


class trainBuilder(setBuilder.builder):
    def datapath(self):
        return train_fileName

    def decode_and_read(self):
        t = RGBImageProcessor.RGBIP(orig_train_picture, gen_train_picture, train_fileName)
        batch = t.read_and_decode()
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            therads = tf.train.start_queue_runners(coord = coord)
            for i in range(t.num_samples):
                example, lab = sess.run(batch)
                #t.show_tensor_to_image(example)
                example = example.reshape(t.imageLength*t.imageLength,3)
                self.training_image_list.append(example)
                self.training_label_list.append(lab)
                if lab not in self.label_list:
                    self.label_list.append(lab)
                self.image_count = self.image_count + 1
#                t.show_gary_tensor_to_image(example)
            coord.request_stop()
            coord.join(therads)
            sess.close()



class testBuilder(setBuilder.builder):
    def datapath(self):
        return test_fileName

    def decode_and_read(self):
        t = RGBImageProcessor.RGBIP(orig_test_picture, gen_test_picture, test_fileName)
        batch = t.read_and_decode()
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            therads = tf.train.start_queue_runners(coord = coord)
            for i in range(t.num_samples):
                example, lab = sess.run(batch)
                #t.show_tensor_to_image(example)
                example = example.reshape(t.imageLength*t.imageLength,3)
                self.training_image_list.append(example)
                self.training_label_list.append(lab)
                if lab not in self.label_list:
                    self.label_list.append(lab)
                self.image_count = self.image_count + 1
#                t.show_gary_tensor_to_image(example)
            coord.request_stop()
            coord.join(therads)
            sess.close()



class validationBuilder(setBuilder.builder):
    def datapath(self):
        return validation_fileName

    def decode_and_read(self):
        t = RGBImageProcessor.RGBIP(orig_validation_picture, gen_validation_picture, validation_fileName)
        batch = t.read_and_decode()
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            therads = tf.train.start_queue_runners(coord = coord)
            for i in range(t.num_samples):
                example, lab = sess.run(batch)
                #t.show_tensor_to_image(example)
                example = example.reshape(t.imageLength*t.imageLength,3)
                self.training_image_list.append(example)
                self.training_label_list.append(lab)
                if lab not in self.label_list:
                    self.label_list.append(lab)
                self.image_count = self.image_count + 1
#                t.show_gary_tensor_to_image(example)
            coord.request_stop()
            coord.join(therads)
            sess.close()
