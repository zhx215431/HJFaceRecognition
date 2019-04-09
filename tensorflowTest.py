import tensorflow as tf;
import numpy as np;
import matplotlib.pyplot as plt;

def conv_k(k,l,result):
    return k < 1

def body_k(k,l,result):
    #result.shape = [1,2*2,2*n,1]
    s = tf.constant(0).get_shape()
    result_l = tf.zeros([1,2,2,1])
    a,b,c= tf.while_loop(conv_l,body_l,[k,l,result_l],shape_invariants=[s,s,tf.TensorShape([None,None,None,None])])
    #e.shape = [1,5,5*M_L_4,1]
    result = tf.cond(tf.cast(k == 0, tf.bool), lambda: c, lambda: tf.concat([result,c],1))
    k = tf.add(k,1)
    return k,l,result

def conv_l(k,l,result):
    return l < 1

def body_l(k,l,result):
    #result.shape = [1,2*n,2,1]
    block_value = [1.1,2.2,3.3,4.4]
    block_value = tf.reshape(block_value,[1,2,2,1])
    result = tf.cond(tf.cast(l == 0, tf.bool), lambda: block_value, lambda: tf.concat([result, block_value],2))
    l = tf.add(l,1)
    return k,l,result

s = tf.constant(0).get_shape()
result_k = tf.zeros([1,2*2,2,1])
a,b,c = tf.while_loop(conv_k,body_k,[0,0,result_k],shape_invariants=[s,s,tf.TensorShape([None,None,None,None])])

sess = tf.InteractiveSession()
print(sess.run(c))
