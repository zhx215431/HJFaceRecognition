import setBuilder
import RGBImageProcessor
import tensorflow as tf
import PIL as Image
import matplotlib.pyplot as plt
import numpy as np

orig_test_picture = 'E:/study/DL/1(test)'
gen_test_picture = 'E:/study/DL/1(test)'
test_fileName = "RGBImage_test.tfrecords"

orig_train_picture = 'E:/study/DL/1.2(train)'
gen_train_picture = 'E:/study/DL/1.2(train)'
train_fileName = "RGBImage_train.tfrecords"

orig_train_high_picture = 'E:/study/DL/1.2(train_high)'
gen_train_high_picture = 'E:/study/DL/1.2(train_high)'
train_high_fileName = "RGBImage_train_high.tfrecords"

orig_train_middle_picture = 'E:/study/DL/1.2(train_middle)'
gen_train_middle_picture = 'E:/study/DL/1.2(train_middle)'
train_middle_fileName = "RGBImage_train_middle.tfrecords"

orig_train_low_picture = 'E:/study/DL/1.2(train_low)'
gen_train_low_picture = 'E:/study/DL/1.2(train_low)'
train_low_fileName = "RGBImage_train_low.tfrecords"

orig_validation_picture = 'E:/study/DL/1(validation)'
gen_validation_picture = 'E:/study/DL/1(validation)'
validation_fileName = "RGBImage_validation.tfrecords"

def make_count_label(input_list):
    count_label = []
    current_number = -9999
    whileCount = 0
    while(whileCount < len(input_list)):
        if (current_number != input_list[whileCount]):
            if whileCount == 0:
                count_label.append(0)
            else:
                count_label.append(whileCount-1)
                count_label.append(whileCount)

        current_number = input_list[whileCount]

        if whileCount == len(input_list) - 1:
            count_label.append(whileCount)
        whileCount = whileCount + 1

    return count_label


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
        self.count_label = make_count_label(self.training_label_list)



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
        self.count_label = make_count_label(self.training_label_list)



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
        self.count_label = make_count_label(self.training_label_list)

class train_highBuilder(setBuilder.builder):
    def datapath(self):
        return train_high_fileName

    def decode_and_read(self):
        t = RGBImageProcessor.RGBIP(orig_train_high_picture, gen_train_high_picture, train_high_fileName)
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
        self.count_label = make_count_label(self.training_label_list)

class train_middleBuilder(setBuilder.builder):
    def datapath(self):
        return train_middle_fileName

    def decode_and_read(self):
        t = RGBImageProcessor.RGBIP(orig_train_middle_picture, gen_train_middle_picture, train_middle_fileName)
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
        self.count_label = make_count_label(self.training_label_list)

class train_lowBuilder(setBuilder.builder):
    def datapath(self):
        return train_low_fileName

    def decode_and_read(self):
        t = RGBImageProcessor.RGBIP(orig_train_low_picture, gen_train_low_picture, train_low_fileName)
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
        self.count_label = make_count_label(self.training_label_list)
