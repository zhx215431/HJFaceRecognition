import tensorflow as tf
import RGBSetBuilder
from tensorflow.python import debug as tfdbg
origin_image_length = 64
M_L_1 = origin_image_length
M_L_2 = origin_image_length//2
M_L_4 = origin_image_length//4
M_L_8 = origin_image_length//8
image_count = 2### TODO:
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
    x_y = neighborhoodDifference_byCombine(base_tensor=X,target_tensor=Y)
    y_x = neighborhoodDifference_byCombine(base_tensor=Y,target_tensor=X)
    return x_y, y_x
'''
neighborhoodDifferences_byCombine
base_tensor     基底张量 shape = [-1,32,32,25] ,基底张量中的值只有0和x
target_tensor   需要与之比较的张量 = [-1,32,32,25] ,目标张量为原张量的片段
output          输出比较结果张量 = [-1,32,32,25,25]
'''
def neighborhoodDifference_byCombine(base_tensor, target_tensor):
    s = tf.constant(0).get_shape()
    result_i = tf.zeros([1,M_L_4*5,M_L_4*5,25])
    a,b,c,d,e,f,g= tf.while_loop(conv_i,body_i,[0,0,0,0,base_tensor,target_tensor,result_i],shape_invariants=[s,s,s,s,base_tensor.get_shape(),target_tensor.get_shape(),tf.TensorShape([None,None,None,None])])
    #g.shape = [X,M_L_4*5,M_L_4*5,25]
    return g
'''
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
                    result_l = tf.cond(l == 0, lambda: value(block_value), lambda: tf.concat([result_l, block_value],2))
                result_k = tf.cond(k == 0, lambda: value(result_l), lambda: tf.concat([result_k,result_l],1))
            result_j = tf.cond(j == 0, lambda: value(result_k), lambda: tf.concat([result_j,result_k],3))
        result_i = tf.cond(i == 0, lambda:value(result_j), lambda: tf.concat([result_i,result_j],0))
    return result_i
'''
def conv_i(i,j,k,l,base_tensor,target_tensor,result):
    return i < image_count - 1

def body_i(i,j,k,l,base_tensor,target_tensor,result):
    #result.shape = [n,M_L_4*5,M_L_4*5,25] n<X
    s = tf.constant(0).get_shape()
    result_j = tf.zeros([1,M_L_4*5,M_L_4*5,1])
    a,b,c,d,e,f,g = tf.while_loop(conv_j,body_j,[i,j,k,l,base_tensor,target_tensor,result_j],shape_invariants=[s,s,s,s,base_tensor.get_shape(),target_tensor.get_shape(),tf.TensorShape([None,None,None,None])])
    #g.shape = [1,M_L_4*5,M_L_4*5,25]
    result = tf.cond(tf.cast(i == 0, tf.bool), lambda:value(g), lambda: tf.concat([result,g],0))
    i = tf.add(1,i)
    return a,b,c,d,base_tensor,target_tensor,result

def conv_j(i,j,k,l,base_tensor,target_tensor,result):
    return j < 25 - 1

def body_j(i,j,k,l,base_tensor,target_tensor,result):
    #result.shape = [1,M_L_4*5,M_L_4*5,n] n<25
    s = tf.constant(0).get_shape()
    result_k = tf.zeros([1,5,M_L_4*5,1])
    a,b,c,d,e,f,g = tf.while_loop(conv_k,body_k,[i,j,0,l,base_tensor,target_tensor,result_k],shape_invariants=[s,s,s,s,base_tensor.get_shape(),target_tensor.get_shape(),tf.TensorShape([None,None,None,None])])
    #g.shape = [1,M_L_4*5,M_L_4*5,1]
    result = tf.cond(tf.cast(j == 0, tf.bool), lambda: value(g), lambda: tf.concat([result,g],3))
    j = tf.add(1,j)
    return a,b,c,d,base_tensor,target_tensor,result

def conv_k(i,j,k,l,base_tensor,target_tensor,result):
    return k < M_L_4 - 1

def body_k(i,j,k,l,base_tensor,target_tensor,result):
    #result.shape = [1,5*n,5*M_L_4,1] n<M_L_4
    s = tf.constant(0).get_shape()
    result_l = tf.zeros([1,5,5,1])
    a,b,c,d,e,f,g = tf.while_loop(conv_l,body_l,[i,j,k,0,base_tensor,target_tensor,result_l],shape_invariants=[s,s,s,s,base_tensor.get_shape(),target_tensor.get_shape(),tf.TensorShape([None,None,None,None])])
    #g.shape = [1,5,5*M_L_4,1]
    result = tf.cond(tf.cast(k == 0, tf.bool), lambda: value(g), lambda: tf.concat([result,g],1))
    k = tf.add(k,1)
    return a,b,c,d,base_tensor,target_tensor,result

def conv_l(i,j,k,l,base_tensor,target_tensor,result):
    return l < M_L_4 - 1

def body_l(i,j,k,l,base_tensor,target_tensor,result):
    #result.shape = [1,5,5*n,1] n<M_L_4
    base_matrix = tf.slice(base_tensor,[i,0,0,j],[1,M_L_4,M_L_4,1])
    target_matrix = tf.slice(target_tensor,[i,0,0,j],[1,M_L_4,M_L_4,1])
    base_matrix = tf.reshape(base_matrix,[M_L_4,M_L_4])
    target_matrix = tf.reshape(target_matrix,[M_L_4,M_L_4])
    block_value = block_neighborhoodDifference(base_matrix,target_matrix,l,k)#返回25个特征值
    block_value = tf.reshape(block_value,[1,5,5,1])
    result = tf.cond(tf.cast(l == 0, tf.bool), lambda: value(block_value), lambda: tf.concat([result, block_value],2))
    l = tf.add(l,1)
    return i,j,k,l,base_tensor,target_tensor,result
'''
value
返回自己
'''
def value(x):
    return x
'''
neighborhoodDifference_bySubstitude
通过替换来获得需要的张量
'''
def neighborhoodDifference_bySubstitude(base_tensor, target_tensor):
    pass
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
    result = tf.cond((x>=2 and x<=M_L_4-3) and (y>=2 and y<=M_L_4-3), block_baseMatrix_not_fillZero(base_matrix, x, y), lambda: block_baseMatrix_should_fillZero(base_matrix, x, y))
'''
block_baseMatrix_should_fillZero
'''
def block_baseMatrix_should_fillZero(base_matrix, x, y):
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
sess = tfdbg.LocalCLIDebugWrapperSession(sess)

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
y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

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
for i in range(30):
    print("start!!!!!!!!!!!!!!")
    base,target,is_same = builder.RE_I_next_batch_image(training_count=2)
    if i%5 == 0:
        train_accuracy = accuracy.eval(feed_dict={x1:base, x2:target, y_:is_same})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x1:base, x2:target, y_:is_same})
