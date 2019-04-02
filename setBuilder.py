import imageProcessor
import tensorflow as tf
import PIL as Image
import matplotlib.pyplot as plt
import numpy as np
import os

#orig_picture = 'E:/study/DL/1(withLabel)'
#gen_picture = 'E:/study/DL/1(withLabel)'
orig_picture = 'E:/study/DL/1(test)'
gen_picture = 'E:/study/DL/1(test)'

#将数字转换成向量 如1->(0,1,0,....,0),4->(0,0,0,0,1,0,0,...,0)
def label_transformer(number, setrange):
    label_array = [];
    for i in range(setrange):
        if i == number:
            label_array.append(1)
        else:
            label_array.append(0)
    return label_array

class builder:
    '''
    @param training_image_list
    @param training_label_list
    @param label_list
    @param image_count
    @param datapath
    '''
    def datapath(self):
        data_path = '子类重写'
    def __init__(self):
        self.training_image_list = []
        self.training_label_list = []
        self.label_list = []
        self.image_count = 0
        self.data_path = self.datapath()

#将训练集读入内存
    def decode_and_read(self):
        print("子类重写")

#随机抽选X个训练
    def next_batch_image(self, training_count):
        whileCount = 0
        random_list = []
        batch_image_list = []
        batch_label_list = []
        while (whileCount < training_count):
            rnd = np.random.randint(1,self.image_count)
            if rnd not in random_list:
                random_list.append(rnd)
                batch_image_list.append(self.training_image_list[rnd - 1])
                batch_label_list.append(label_transformer(number=self.training_label_list[rnd - 1],setrange=len(self.label_list)))
#                t = imageProcessor.IP(orig_picture=orig_picture, gen_picture=gen_picture)
#                print(self.image_count)
#                print(rnd)
#                print(label_transformer(number=self.training_label_list[rnd - 1],setrange=len(self.label_list)))
#                tes = self.training_image_list[rnd - 1]
#                tes = tes.reshape(64,64,3)
#                t.show_tensor_to_image(tes)
#                os.system('pause')
                whileCount = whileCount + 1
        return batch_image_list, batch_label_list

#人脸重识别的随机抽取x个组合训练
    def RE_I_next_batch_image(self, training_count):
        base_image_list = []
        target_image_list = []
        is_same_list = []
        for i in range(training_count):
            rnd1 = np.random.randint(1,self.image_count)
            rnd2 = np.random.randint(1,self.image_count)
            base_image_list.append(self.training_image_list[rnd1 - 1])
            target_image_list.append(self.training_image_list[rnd2 - 1])
            if self.training_label_list[rnd1 - 1] == self.training_label_list[rnd2 - 1]:
                is_same_list.append(1)
            else:
                is_same_list.append(0)
        return base_image_list, target_image_list, is_same_list

#随机抽选X个测试
    def test_batch_image(self, test_count):
        whileCount = 0
        random_list = []
        batch_image_list = []
        batch_label_list = []
        while (whileCount < test_count):
            rnd = np.random.randint(1,self.image_count)
            if rnd not in random_list:
                random_list.append(rnd)
                batch_image_list.append(self.training_image_list[rnd - 1])
                batch_label_list.append(self.training_label_list[rnd - 1])
                whileCount = whileCount + 1
                return batch_image_list, batch_label_list
