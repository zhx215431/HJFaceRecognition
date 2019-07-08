'''
以柱状图的形式画出训练集（或验证集或测试集）的每个class数量图
'''
import matplotlib.pyplot as plt
import numpy as np
import RGBSetBuilder

def draw_train_set():
    setBuilder = RGBSetBuilder.trainBuilder()
    setBuilder.decode_and_read()
    y_list = []
    for i in range(int(len(setBuilder.count_label)/2)):
        count = setBuilder.count_label[1 + i * 2] - setBuilder.count_label[i * 2] + 1
        y_list.append(count)
    plt.figure(figsize=(8,6), dpi=80)#8*6的窗口 分辨率为80像素/英寸
    plt.subplot(1,1,1)#子图
    N = len(setBuilder.label_list)
    index = np.arange(N)
    width = 0.35
    p = plt.bar(index,y_list,width,label="distribution",color="#87CEFA")


    plt.legend(loc="upper right")
    plt.show()

draw_train_set()
