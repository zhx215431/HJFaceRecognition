import tensorflow as tf
import RGBSetBuilder
from tensorflow.python import debug as tfdbg
origin_image_length = 64
M_L_1 = origin_image_length
M_L_2 = origin_image_length//2
M_L_4 = origin_image_length//4
M_L_8 = origin_image_length//8
image_count = 50### TODO:
'''
权重初始化
shape           权重形状
name            node名称
output          该形状的权重张量
'''
def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.1, name='weight_init')
    return tf.Variable(initial,name=name)
'''
偏置初始化
shape           偏置形状
name            node名称
output          该形状的偏置张量
'''
def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape, name = 'bias_init')
    return tf.Variable(initial,name=name)
'''
卷积
x               被积张量
W               卷积核
stride          卷积步长
name            node名称
output          结果张量
'''
def conv2d(x,W,stride,name):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME',name=name)
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
base_tensor     基底张量 shape = [-1,M_L_4,M_L_4,25] ,基底张量中的值只有0和x
target_tensor   需要与之比较的张量 = [-1,M_L_4,M_L_4,25] ,目标张量为原张量的片段
output          输出比较结果张量 = [-1,M_L_4*5,M_L_4*5,25]
'''
def neighborhoodDifference_byCombine(base_tensor, target_tensor):
    result = tf.zeros([1,5,5,1])
    a,b,c,d,e,f,g = tf.while_loop(cond=cond_i,body=body_i,loop_vars=[0,0,0,0,result,base_tensor,target_tensor],shape_invariants=[sShape(),sShape(),sShape(),sShape(),tf.TensorShape([None,None,None,None]),base_tensor.get_shape(),target_tensor.get_shape()])
    return e

def cond_i(i,j,k,l,result,base_tensor, target_tensor):
    return tf.less(i,image_count)

def cond_j(i,j,k,l,result,base_tensor, target_tensor):
    return tf.less(j,M_L_4)

def cond_k(i,j,k,l,result,base_tensor, target_tensor):
    return tf.less(k,M_L_4)

def cond_l(i,j,k,l,result,base_tensor, target_tensor):
    return tf.less(l,25)

def body_i(i,j,k,l,result,base_tensor, target_tensor):
    a,b,c,d,e,f,g = tf.while_loop(cond=cond_j,body=body_j,loop_vars=[i,0,k,l,result,base_tensor, target_tensor],shape_invariants=[sShape(),sShape(),sShape(),sShape(),tf.TensorShape([None,None,None,None]),base_tensor.get_shape(),target_tensor.get_shape()])
    result = tf.cond(tf.cast(tf.equal(i,0), tf.bool), lambda:e, lambda: tf.concat([result,e],0))
    i = tf.add(i,1)
    return i,j,k,l,result,base_tensor, target_tensor

def body_j(i,j,k,l,result,base_tensor, target_tensor):
    a,b,c,d,e,f,g = tf.while_loop(cond=cond_k,body=body_k,loop_vars=[i,j,0,l,result,base_tensor, target_tensor],shape_invariants=[sShape(),sShape(),sShape(),sShape(),tf.TensorShape([None,None,None,None]),base_tensor.get_shape(),target_tensor.get_shape()])
    result = tf.cond(tf.cast(tf.equal(j,0), tf.bool), lambda: e, lambda: tf.concat([result,e],1))
    j = tf.add(j,1)
    return i,j,k,l,result,base_tensor, target_tensor

def body_k(i,j,k,l,result,base_tensor, target_tensor):
    a,b,c,d,e,f,g = tf.while_loop(cond=cond_l,body=body_l,loop_vars=[i,j,k,0,result,base_tensor, target_tensor],shape_invariants=[sShape(),sShape(),sShape(),sShape(),tf.TensorShape([None,None,None,None]),base_tensor.get_shape(),target_tensor.get_shape()])
    result = tf.cond(tf.cast(tf.equal(k,0), tf.bool), lambda: e, lambda: tf.concat([result,e],2))
    k = tf.add(k,1)
    return i,j,k,l,result,base_tensor, target_tensor

def body_l(i,j,k,l,result,base_tensor, target_tensor):
    base_matrix = tf.slice(base_tensor,[i,0,0,l],[1,M_L_4,M_L_4,1])
    target_matrix = tf.slice(target_tensor,[i,0,0,l],[1,M_L_4,M_L_4,1])
    base_matrix = tf.reshape(base_matrix,[M_L_4,M_L_4])
    target_matrix = tf.reshape(target_matrix,[M_L_4,M_L_4])
    block_value = block_neighborhoodDifference(base_matrix,target_matrix,k,j)#返回25个特征值
    block_value = tf.reshape(block_value,[1,5,5,1])
    #block_value = tf.ones([1,5,5,1])
    result = tf.cond(tf.cast(tf.equal(l,0), tf.bool), lambda: block_value, lambda: tf.concat([result, block_value],3))
    l = tf.add(l,1)
    return i,j,k,l,result,base_tensor, target_tensor

def sShape():
    return tf.constant(0).get_shape()
'''
neighborhoodDifference_bySubstitude
通过替换来获得需要的张量
'''
def neighborhoodDifference_bySubstitude(base_tensor, target_tensor):
    pass
'''
block_neighborhoodDifference
base_matrix     基底矩阵 shape = [M_L_4,M_L_4]
target_matrix   需要与之比较的矩阵 shape = [M_L_4,M_L_4]
x               矩阵横坐标
y               矩阵纵坐标
output          [a,b....]共25个值
'''
def block_neighborhoodDifference(base_matrix, target_matrix, x, y):
    base_not_fillZero = block_baseMatrix_not_fillZero(base_matrix, x, y)
    target_fillZero = block_targetMatrix_fillZero(target_matrix, x, y)
    base_not_fillZero = tf.reshape(base_not_fillZero,[25])
    target_fillZero = tf.reshape(target_fillZero,[25])
    result = tf.subtract(base_not_fillZero,target_fillZero)
    return result

'''
block_baseMatrix_should_fillZero
base_matrix   基底矩阵 shape = [M_L_4,M_L_4]
x             矩阵横坐标
y             矩阵纵坐标
output        f(x,y)l(5,5) 边界填零 (列表形式，25个值)
'''
def block_baseMatrix_should_fillZero(base_matrix, x, y):
    value = base_matrix[y][x]
    matrix = tf.fill([M_L_4,M_L_4],value)
    matrix = tf.pad(matrix,[[2,2],[2,2]],"CONSTANT")#边界填零
    result = tf.slice(matrix,[y,x],[5,5])
    return result
'''
block_baseMatrix_not_fillZero
base_matrix    基底矩阵 shape = [M_L_4,M_L_4]
x              矩阵横坐标
y              矩阵纵坐标
output         f(x,y)l(5,5) 边界不填零 (列表形式，25个值)
'''
def block_baseMatrix_not_fillZero(base_matrix, x, y):
    value = base_matrix[y][x]
    result = tf.fill([25],value)
    return result

'''
block_targetMatrix_fillZero
target_matrix  目标矩阵 shape = [M_L_4,M_L_4]
x              矩阵heng坐标
y              矩阵zong坐标
output         N[g(x,y)] 边界填零 (列表形式，25个值)
'''
def block_targetMatrix_fillZero(target_matrix, x, y):
    matrix = tf.pad(target_matrix,[[2,2],[2,2]],"CONSTANT")
    result = tf.slice(matrix,[y,x],[5,5])
    result = tf.reshape(result,[25])
    return result

'''
池化
x               需要池化的张量
poolSize        池化窗口宽度，池化步长
name            node名称
output          池化结果张量
'''
def max_pool(x,poolSize,name):
    return tf.nn.max_pool(x, ksize=[1,poolSize,poolSize,1], strides=[1,poolSize,poolSize,1], padding='SAME',name=name)
'''
合并两个张量
X              张量1,shape = [-1, M_L_8, M_L_8, 25]
Y              张量2,shape = [-1, M_L_8, M_L_8, 25]
output         合并后的张量,shape = [-1 , M_L_8, M_L_8, 50]
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
x1 = tf.placeholder("float",shape=[None,M_L_1*M_L_1,3],name = 'x1')
x2 = tf.placeholder("float",shape=[None,M_L_1*M_L_1,3],name = 'x2')
y_ = tf.placeholder("float",shape=[None,2],name='y_')

'''
第一层卷积 M_L_1*M_L_1*3 -> M_L_2*M_L_2*20
'''
W_conv1 = weight_variable([5,5,3,20],'W_conv1')
b_conv1 = bias_variable([20],'b_conv1')
x1_image = tf.reshape(x1,[-1,M_L_1,M_L_1,3],name='x1_image_reshape')
x2_image = tf.reshape(x2,[-1,M_L_1,M_L_1,3],name='x2_image_reshape')
hx1_conv1 = conv2d(x=x1_image, W=W_conv1, stride=1, name='hx1_conv1')
hx2_conv1 = conv2d(x=x2_image, W=W_conv1, stride=1, name='hx2_conv1')
hx1_conv1 = tf.add(hx1_conv1,b_conv1,name='hx1_conv1_bias')
hx2_conv1 = tf.add(hx2_conv1,b_conv1,name='hx2_conv1_bias')
hx1_pool1 = max_pool(hx1_conv1,2,'hx1_pool1')
hx2_pool1 = max_pool(hx2_conv1,2,'hx2_pool1')

'''
第二层卷积 M_L_2*M_L_2*20 -> M_L_4*M_L_4*25
'''
W_conv2 = weight_variable([5,5,20,25],'W_conv2')
b_conv2 = bias_variable([25],'b_conv2')
hx1_conv2 = conv2d(x=hx1_pool1, W=W_conv2, stride=1, name='hx1_conv2')
hx2_conv2 = conv2d(x=hx2_pool1, W=W_conv2, stride=1, name='hx2_conv2')
hx1_conv2 = tf.add(hx1_conv2,b_conv2,name='hx1_conv2_bias')
hx2_conv2 = tf.add(hx2_conv2,b_conv2,name='hx2_conv2_bias')
hx1_pool2 = max_pool(hx1_conv2,2,'hx1_pool2')
hx2_pool2 = max_pool(hx2_conv2,2,'hx2_pool2')

'''
cross input neighborhood Differences
2*[M_L_4*M_L_4*25] -> 2*[(M_L_4*5) * (M_L_4*5) * 25 ]
'''
x_y_neighborhoodDiff, y_x_neighborhoodDiff = cross_input_neighborhoodDifferences(X=hx1_pool2, Y=hx2_pool2)
x_y_conv1 = tf.nn.relu(x_y_neighborhoodDiff)
y_x_conv1 = tf.nn.relu(y_x_neighborhoodDiff)

'''
Patch Summary Features
2*[(M_L_4*5) * (M_L_4*5) * 25] -> 2*[M_L_4*M_L_4*25]
'''
W_x_y_conv1 = weight_variable([5,5,25,25],'W_x_y_conv1')
W_y_x_conv1 = weight_variable([5,5,25,25],'W_y_x_conv1')
b_x_y_conv1 = bias_variable([25],'b_x_y_conv1')
b_y_x_conv1 = bias_variable([25],'b_y_x_conv1')
h_x_y_relu1 = tf.nn.relu(conv2d(x=x_y_conv1, W=W_x_y_conv1, stride=5, name='h_x_y_relu1') + b_x_y_conv1)
h_y_x_relu1 = tf.nn.relu(conv2d(x=y_x_conv1, W=W_y_x_conv1, stride=5, name='h_y_x_relu1') + b_y_x_conv1)

'''
Across Patch Features
2*[M_L_4*M_L_4*25] -> 2*[M_L_8*M_L_8*25]
'''
W_x_y_conv2 = weight_variable([3,3,25,25],'W_x_y_conv2')
W_y_x_conv2 = weight_variable([3,3,25,25],'W_y_x_conv2')
b_x_y_conv2 = bias_variable([25],'b_x_y_conv2')
b_y_x_conv2 = bias_variable([25],'b_y_x_conv2')
h_x_y_conv = conv2d(x=h_x_y_relu1, W=W_x_y_conv2, stride=1, name='h_x_y_conv') + b_x_y_conv2
h_y_x_conv = conv2d(x=h_y_x_relu1, W=W_y_x_conv2, stride=1, name='h_y_x_conv') + b_y_x_conv2#[2,M_L_4,M_L_4,25]
h_x_y_pool = max_pool(x=h_x_y_conv, poolSize=2, name='h_x_y_pool')
h_y_x_pool = max_pool(x=h_y_x_conv, poolSize=2, name='h_y_x_pool')

'''
Higher-Order Relationships
2*[M_L_8*M_L_8*25] -> [M_L_8*M_L_8*50] ->500
'''
combine_x_y = combineTensor(X=h_x_y_pool,Y=h_y_x_pool)
combine_x_y_flat = tf.reshape(combine_x_y, [-1,M_L_8*M_L_8*50])
W_fc1 = weight_variable([M_L_8*M_L_8*50, 500],'W_fc1')
b_fc1 = bias_variable([500],'b_fc1')
h_fc1 = tf.nn.relu(tf.matmul(combine_x_y_flat, W_fc1) + b_fc1)

'''
DROUP OUT
## TODO:
'''
'''
输出层
500 -> 2
'''
W_fc2 = weight_variable([500,2],'W_fc2')
b_fc2 = bias_variable([2],'b_fc2')
y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2, name = 'y_conv')

'''
损失函数
'''
## TODO:
#y_conv = tf.slice(y_conv,[0,0],[50,2])
cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-8,1.0)))
'''
训练模型&模型评估
'''

train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.global_variables_initializer())

'''
debug
'''
#base,target,is_same = builder.RE_I_next_batch_image(training_count=2)
sess = tfdbg.LocalCLIDebugWrapperSession(sess)
#print(sess.run(h_y_x_pool,feed_dict={x1:base, x2:target, y_:is_same}))



for i in range(300):
    print("start!!!!!!!!!!!!!!")
    base,target,is_same = builder.RE_I_next_batch_image(training_count=50)
    if i%1 == 0:
        train_cross_entropy = cross_entropy.eval(feed_dict={x1:base, x2:target, y_:is_same})
        print("step %d, cross entropy: %g"%(i, train_cross_entropy))
        #print(x_y_neighborhoodDiff.eval(feed_dict={x1:base, x2:target, y_:is_same}))
    train_step.run(feed_dict={x1:base, x2:target, y_:is_same})
