'''
把训练过程中的训练集，验证集的准确率和交叉熵写入excel中
'''
import xlwt
import xlrd
import xlutils
from xlutils.copy import copy

class write():
    '''
    @param datapath 工作表地址
    @param data 工作表
    @param index 工作表索引
    '''
    def __init__(self,data_path,index):
        self.datapath = data_path
        self.data = copy(xlrd.open_workbook(self.datapath))
        self.table = self.data.get_sheet(index)

#适用于test1,test2,test3,test4
    def write_append(self,row,value1,value2,value3,value4):
        self.table.write(row,0,str(value1))
        self.table.write(row,1,str(value2))
        self.table.write(row,2,str(value3))
        self.table.write(row,3,str(value4))
        self.data.save(self.datapath)

    def write_exappend(self,row,value5,value6,value7,value8,value9,value10):
        self.table.write(row,4,str(value5))
        self.table.write(row,5,str(value6))
        self.table.write(row,6,str(value7))
        self.table.write(row,7,str(value8))
        self.table.write(row,8,str(value9))
        self.table.write(row,9,str(value10))


    def test(self):
        print(self.datapath)
        print(self.data)
        print(self.table)
