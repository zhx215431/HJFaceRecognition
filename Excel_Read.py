'''
在excel中读取训练集验证集相关数据并显示在一张图标上
'''
import xlrd
import matplotlib.pyplot as plt
excelFilePath = 'E:/study/DL/HJFaceRecognition/project/ExcelR&W/test.xls'


def translation_accuracy():
    data = xlrd.open_workbook(excelFilePath)


    table = data.sheets()[5]


    rows = table.nrows
    train_accuracy_list = []
    validation_accuracy_list = []
    time_list = []
    for i in range(rows - 1):
        train_accuracy = table.cell_value(i+1,0)
        validation_accuracy = table.cell_value(i+1,2)

        train_accuracy_list.append(float(train_accuracy))
        validation_accuracy_list.append(float(validation_accuracy))

        time_list.append(i*5)
    plt.title('accuracy')
    plt.plot(time_list, train_accuracy_list, color='green', label='train accuracy')
    plt.plot(time_list, validation_accuracy_list, color='red', label='validation accuracy')
    plt.legend()

    plt.xlabel('time')
    plt.ylabel('rate')
    plt.show()

def translation_cross_entropy():
    data = xlrd.open_workbook(excelFilePath)


    table = data.sheets()[5]


    rows = table.nrows
    train_cross_entropy_list = []
    validation_cross_entropy_list = []
    time_list = []
    for i in range(rows - 1):
        train_cross_entropy = table.cell_value(i+1,1)
        validation_cross_entropy = table.cell_value(i+1,3)

        train_cross_entropy_list.append(float(train_cross_entropy))
        validation_cross_entropy_list.append(float(validation_cross_entropy))

        time_list.append(i*5)
    plt.title('cross_entropy')
    plt.plot(time_list, train_cross_entropy_list, color='green', label='train cross entropy')
    plt.plot(time_list, validation_cross_entropy_list, color='red', label='validation cross entropy')
    plt.legend()

    plt.xlabel('time')
    plt.ylabel('cross entropy')
    plt.show()

def translation_accuracy_ex():
    pass

def translation_cross_entropy_ex():
    pass
