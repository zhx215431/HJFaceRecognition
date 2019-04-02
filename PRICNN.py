import tensorflow as tf
import RGBSetBuilder
from tensorflow.python import debug as tfdbg
origin_image_length = 128
M_L_1 = origin_image_length
M_L_2 = origin_image_length//2
M_L_4 = origin_image_length//4
M_L_8 = origin_image_length//8
image_count = 500### TODO:
'''
权重初始化
shape           权重形状
output          该形状的权重张量
'''
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
'''
偏置初始化
shape           偏置形状
output          该形状的偏置张量
'''
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
'''
卷积
x               被积张量
W               卷积核
stride          卷积步长
output          结果张量
'''
def conv2d(x,W,stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
'''
cross_input_neighborhoodDifferences
X               输入张量1
Y               输入张量2
output          近邻比较后的两个张量
'''
def cross_input_neighborhoodDifferences(X,Y):
    x_y = neighborhoodDifference(base_tensor=X,target_tensor=Y)
    y_x = neighborhoodDifference(base_tensor=Y,target_tensor=X)
    return x_y, y_x
'''
neighborhood Differences
base_tensor     基底张量 shape = [-1,32,32,25]
target_tensor   需要与之比较的张量 = [-1,32,32,25]
output          输出比较结果张量 = [-1,32,32,25,25]
'''
def neighborhoodDifference(base_tensor, target_tensor):
    oriShape = tf.shape(base_tensor)
    count = oriShape[0]
    result_l = tf.zeros([1,1,1,1,25])
    result_k = tf.zeros([1,1,M_L_4,1,25])
    result_j = tf.zeros([1,M_L_4,M_L_4,1,25])
    result_i = tf.zeros([1,M_L_4,M_L_4,25,25])
    for i in range(image_count):
        ##第几张图片
        print("开始处理第%d张图片" %(i))
        for j in range(25):
            ##第几个特征值
            print("开始处理第%d个特征值" %(j))
            for k in range(M_L_4):
                ##高度遍历
                for l in range(M_L_4):
                    ##宽度遍历
                    base_matrix = tf.slice(base_tensor,[i,0,0,j],[1,M_L_4,M_L_4,1])
                    target_matrix = tf.slice(target_tensor,[i,0,0,j],[1,M_L_4,M_L_4,1])
                    base_matrix = tf.reshape(base_matrix,[M_L_4,M_L_4])
                    target_matrix = tf.reshape(target_matrix,[M_L_4,M_L_4])
                    block_value = block_neighborhoodDifference(base_matrix,target_matrix,l,k)#返回25个特征值
                    block_value = tf.reshape(block_value,[1,1,1,1,25])
                    if l == 0:
                        result_l = block_value
                    else:
                        result_l = tf.concat([result_l, block_value],2)
                if k == 0:
                    result_k = result_l
                else:
                    result_k = tf.concat([result_k,result_l],1)
            if j == 0:
                result_j = result_k
            else:
                result_j = tf.concat([result_j,result_k],3)
        if i == 0:
            result_i = result_j
        else:
            result_i = tf.concat([result_i,result_j],0)
    return result_i
'''
block_neighborhoodDifference
base_matrix     基底矩阵 shape = [32,32]
target_matrix   需要与之比较的矩阵 shape = [32,32]
x               矩阵横坐标
y               矩阵纵坐标
output          [a,b....]共25个值
'''
def block_neighborhoodDifference(base_matrix, target_matrix, x, y):
#    base_fillZero = block_baseMatrix_fillZero(base_matrix, x, y)
    base_not_fillZero = block_baseMatrix_not_fillZero(base_matrix, x, y)
    target_fillZero = block_targetMatrix_fillZero(target_matrix, x, y)
    base_not_fillZero = tf.reshape(base_not_fillZero,[25])
    target_fillZero = tf.reshape(target_fillZero,[25])
    result = tf.subtract(base_not_fillZero,target_fillZero)
    return result
'''
block_baseMatrix_fillZero
base_matrix    基底矩阵 shape = [32,32]
x              矩阵横坐标
y              矩阵纵坐标
output         f(x,y)l(5,5) 边界填零 (列表形式，25个值)
'''
def block_baseMatrix_fillZero(base_matrix, x, y):
    if (x>=2 and x<=M_L_4-3) and (y>=2 and y<=M_L_4-3):
        return block_baseMatrix_not_fillZero(base_matrix, x, y)
    else:
        value = base_matrix[x][y]
        matrix = tf.fill([32,32],value)
        matrix = tf.pad(matrix,[[2,2],[2,2]],"CONSTANT")#边界填零
        result = tf.slice(matrix,[x,y],[5,5])
        result = tf.reshape(result,[25])
        return result

'''
block_baseMatrix_not_fillZero
base_matrix    基底矩阵 shape = [32,32]
x              矩阵横坐标
y              矩阵纵坐标
output         f(x,y)l(5,5) 边界不填零 (列表形式，25个值)
'''
def block_baseMatrix_not_fillZero(base_matrix, x, y):
    value = base_matrix[x][y]
    result = tf.fill([25],value)
    return result

'''
block_targetMatrix_fillZero
target_matrix  目标矩阵 shape = [32,32]
x              矩阵横坐标
y              矩阵纵坐标
output         N[g(x,y)] 边界填零 (列表形式，25个值)
'''
def block_targetMatrix_fillZero(target_matrix, x, y):
    matrix = tf.pad(target_matrix,[[2,2],[2,2]],"CONSTANT")
    result = tf.slice(matrix,[x,y],[5,5])
    result = tf.reshape(result,[25])
    return result

'''
池化
x               需要池化的张量
poolSize        池化窗口宽度，池化步长
output          池化结果张量
'''
def max_pool(x,poolSize):
    return tf.nn.max_pool(x, ksize=[1,poolSize,poolSize,1], strides=[1,poolSize,poolSize,1], padding='SAME')
'''
合并两个张量
X              张量1,shape = [-1, 16, 16, 25]
Y              张量2,shape = [-1, 16, 16, 25]
output         合并后的张量,shape = [-1 , 16, 16, 50]
'''
def combineTensor(X,Y):
    combine = tf.concat([X,Y], 3)
    return combine

builder = RGBSetBuilder.RGBSetBuilder()
builder.decode_and_read()
print(builder.image_count)
print(builder.label_list)
print(builder.training_label_list)
sess = tf.InteractiveSession()

'''
占位符
'''
x1 = tf.placeholder("float",shape=[None,M_L_1*M_L_1,3])
x2 = tf.placeholder("float",shape=[None,M_L_1*M_L_1,3])
y_ = tf.placeholder("float",shape=[None,2])

'''
第一层卷积 128*128*3 -> 64*64*20
'''
W_conv1 = weight_variable([5,5,3,20])
b_conv1 = bias_variable([20])
x1_image = tf.reshape(x1,[-1,M_L_1,M_L_1,3])
x2_image = tf.reshape(x2,[-1,M_L_1,M_L_1,3])
hx1_conv1 = conv2d(x=x1_image, W=W_conv1, stride=1) + b_conv1
hx2_conv1 = conv2d(x=x2_image, W=W_conv1, stride=1) + b_conv1
hx1_pool1 = max_pool(hx1_conv1,2)
hx2_pool1 = max_pool(hx2_conv1,2)

'''
第二层卷积 64*64*20 -> 32*32*25
'''
W_conv2 = weight_variable([5,5,20,25])
b_conv2 = bias_variable([25])
hx1_conv2 = conv2d(x=hx1_pool1, W=W_conv2, stride=1) + b_conv2
hx2_conv2 = conv2d(x=hx2_pool1, W=W_conv2, stride=1) + b_conv2
hx1_pool2 = max_pool(hx1_conv2,2)
hx2_pool2 = max_pool(hx2_conv2,2)

'''
cross input neighborhood Differences
2*[32*32*25] -> 2*[(32*5) * (32*5) * 25 ]
'''
x_y_neighborhoodDiff, y_x_neighborhoodDiff = cross_input_neighborhoodDifferences(X=hx1_pool2, Y=hx2_pool2)
x_y_conv1 = tf.nn.relu(x_y_neighborhoodDiff)
y_x_conv1 = tf.nn.relu(y_x_neighborhoodDiff)

'''
Patch Summary Features
2*[(32*5) * (32*5) * 25] -> 2*[32*32*25]
'''
W_x_y_conv1 = weight_variable([5,5,25,25])
W_y_x_conv1 = weight_variable([5,5,25,25])
b_x_y_conv1 = bias_variable([25])
b_y_x_conv1 = bias_variable([25])
h_x_y_relu1 = tf.nn.relu(conv2d(x=x_y_conv1, W=W_x_y_conv1, stride=5) + b_x_y_conv1)
h_y_x_relu1 = tf.nn.relu(conv2d(x=y_x_conv1, W=W_y_x_conv1, stride=5) + b_y_x_conv1)

'''
Across Patch Features
2*[32*32*25] -> 2*[16*16*25]
'''
W_x_y_conv2 = weight_variable([3,3,25,25])
W_y_x_conv2 = weight_variable([3,3,25,25])
b_x_y_conv2 = bias_variable([25])
b_y_x_conv2 = bias_variable([25])
h_x_y_conv = conv2d(x=h_x_y_relu1, W=W_x_y_conv2, stride=1) + b_x_y_conv2
h_y_x_conv = conv2d(x=h_y_x_relu1, W=W_y_x_conv2, stride=1) + b_y_x_conv2
h_x_y_pool = max_pool(x=h_x_y_conv, poolSize=2)
h_y_x_pool = max_pool(x=h_y_x_conv, poolSize=2)

'''
Higher-Order Relationships
2*[16*16*25] -> [16*16*50] ->500
'''
combine_x_y = combineTensor(X=h_x_y_pool,Y=h_y_x_pool)
combine_x_y_flat = tf.reshape(combine_x_y, [-1,M_L_8*M_L_8*50])
W_fc1 = weight_variable([M_L_8*M_L_8*50, 500])
b_fc1 = bias_variable([500])
h_fc1 = tf.nn.relu(tf.matmul(combine_x_y_flat, W_fc1) + b_fc1)

'''
DROUP OUT
不知道要不要
'''
'''
输出层
500 -> 2
'''
W_fc2 = weight_variable([500,2])
b_fc2 = bias_variable([2])
y_conv = ft.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

'''
损失函数
'''
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
'''
训练模型&模型评估
'''
train_step = tf.train.AdamOptimizer(1e-2).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.global_variables_initializer())
for i in range(20000):
    print("start!!!!!!!!!!!!!!")
    base,target,is_same = builder.RE_I_next_batch_image(training_count=50)
    if i%5 == 0:
        train_accuracy = accuracy.eval(feed_dict={x1:base, x2:target, y_:is_same})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x1:base, x2:target, y_:is_same})
