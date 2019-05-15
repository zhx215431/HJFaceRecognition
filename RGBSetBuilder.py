import setBuilder
import RGBImageProcessor
import tensorflow as tf
import PIL as Image
import matplotlib.pyplot as plt
import numpy as np

#orig_picture = 'E:/study/DL/1(withLabel)'
#gen_picture = 'E:/study/DL/1(withLabel)'
orig_picture = 'E:/study/DL/1(test)'
gen_picture = 'E:/study/DL/1(test)'


class RGBSetBuilder(setBuilder.builder):
    def datapath(self):
        filename = "RGBImage.tfrecords"
        return filename
#将训练集读入内存
    def decode_and_read(self):
        t = RGBImageProcessor.RGBIP(orig_picture, gen_picture)
        #t.read_image()
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
