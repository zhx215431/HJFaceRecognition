#import tensorflowWhileLoop
import tensorflow as tf
import RGBSetBuilder
import Excel_Write
import Excel_Read

origin_image_length = 64
M_L_1 = origin_image_length
M_L_2 = origin_image_length//2
M_L_4 = origin_image_length//4
M_L_8 = origin_image_length//8
#whileLoop = tensorflowWhileLoop.tensorflow_while_loop(None,None,2)
#result = whileLoop.neighborhoodDifference()
#shape = tf.shape(result)

def block_value(i,j,k,l,base_tensor,target_tensor):
    base_matrix = tf.slice(base_tensor,[i,0,0,l],[1,M_L_4,M_L_4,1])
    target_matrix = tf.slice(target_tensor,[i,0,0,l],[1,M_L_4,M_L_4,1])
    base_matrix = tf.reshape(base_matrix,[M_L_4,M_L_4])
    target_matrix = tf.reshape(target_matrix,[M_L_4,M_L_4])
    block_value = block_neighborhoodDifference(base_matrix,target_matrix,k,j)#返回25个特征值
    block_value = tf.reshape(block_value,[1,5,5,1])
    return block_value

def block_neighborhoodDifference(base_matrix, target_matrix, x, y):
    base_not_fillZero = block_baseMatrix_not_fillZero(base_matrix, x, y)
    target_fillZero = block_targetMatrix_fillZero(target_matrix, x, y)
    base_not_fillZero = tf.reshape(base_not_fillZero,[25])
    target_fillZero = tf.reshape(target_fillZero,[25])
    result = tf.subtract(base_not_fillZero,target_fillZero)
    return result

def block_baseMatrix_not_fillZero(base_matrix, x, y):
    value = base_matrix[y][x]
    result = tf.fill([25],value)
    return result

def block_targetMatrix_fillZero(target_matrix, x, y):
    matrix = tf.pad(target_matrix,[[2,2],[2,2]],"CONSTANT")
    result = tf.slice(matrix,[y,x],[5,5])
    result = tf.reshape(result,[25])
    return result

#Excel_Read.translation_accuracy()
#Excel_Read.translation_cross_entropy()


a = [[[1,2,],[3,4]],[[5,6],[7,8]],[[9,0],[1,2]]]
t = tf.shape(a)
sess = tf.Session()
print(sess.run(t))
