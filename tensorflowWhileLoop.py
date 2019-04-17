import tensorflow as tf;
import numpy as np;
import matplotlib.pyplot as plt;
image_count = 2


def ywwuyi():
    return tf.ones([1,5,5,1])

def ywwuyi_2(i,j,k,l):
    t = tf.fill([1,5,5,1],l)
    t = tf.cast(t,tf.float32)
    return t
#i -> batch
#j -> height
#k -> width
#l -> channel
def cond_test(i,j,k,l,result):
    return True

def cond_i(i,j,k,l,result):
    return tf.less(i,image_count)

def cond_j(i,j,k,l,result):
    return tf.less(j,32)

def cond_k(i,j,k,l,result):
    return tf.less(k,32)

def cond_l(i,j,k,l,result):
    return tf.less(l,25)

def body_i(i,j,k,l,result):
    a,b,c,d,e = tf.while_loop(cond=cond_j,body=body_j,loop_vars=[i,0,k,l,result])
    result = tf.cond(tf.cast(tf.equal(i,0), tf.bool), lambda:e, lambda: tf.concat([result,e],0))
    i = tf.add(i,1)
    return i,j,k,l,result

def body_j(i,j,k,l,result):
    a,b,c,d,e = tf.while_loop(cond=cond_k,body=body_k,loop_vars=[i,j,0,l,result],shape_invariants=[sShape(),sShape(),sShape(),sShape(),tf.TensorShape([None,None,None,None])])
    result = tf.cond(tf.cast(tf.equal(j,0), tf.bool), lambda: e, lambda: tf.concat([result,e],1))
    j = tf.add(j,1)
    return i,j,k,l,result

def body_k(i,j,k,l,result):
    a,b,c,d,e = tf.while_loop(cond=cond_l,body=body_l,loop_vars=[i,j,k,0,result],shape_invariants=[sShape(),sShape(),sShape(),sShape(),tf.TensorShape([None,None,None,None])])
    result = tf.cond(tf.cast(tf.equal(k,0), tf.bool), lambda: e, lambda: tf.concat([result,e],2))
    k = tf.add(k,1)
    return i,j,k,l,result

def body_l(i,j,k,l,result):
    result = tf.cond(tf.cast(tf.equal(l,0), tf.bool), lambda: ywwuyi_2(i,j,k,l), lambda: tf.concat([result, ywwuyi_2(i,j,k,l)],3))
    l = tf.add(l,1)
    return i,j,k,l,result

def sShape():
    return tf.constant(0).get_shape()

class tensorflow_while_loop:
    """docstring for tensorflow_while_loop."""
    def __init__(self,base_tensor,target_tensor,image_count):
        self.base_tensor = base_tensor
        self.target_tensor = target_tensor

    def neighborhoodDifference(self):
        result = tf.zeros([1,5,5,1])
        a,b,c,d,e = tf.while_loop(cond=cond_i,body=body_i,loop_vars=[0,0,0,0,result],shape_invariants=[sShape(),sShape(),sShape(),sShape(),tf.TensorShape([None,None,None,None])])
        return e
#[1,5,5,1] -> [-1,32*5,32*5,25]
'''
result = tf.zeros([1,5,5,1])
s = tf.constant(0).get_shape()
a,b,c,d,e = tf.while_loop(cond=cond_i,body=body_i,loop_vars=[0,0,0,0,result],shape_invariants=[s,s,s,s,tf.TensorShape([None,None,None,None])])
f = tf.shape(e)
sess = tf.InteractiveSession()
print(sess.run(e))#result
print(sess.run(a))#i
print(sess.run(b))#j
print(sess.run(c))#k
print(sess.run(d))#l
print(sess.run(f))#shape
'''
