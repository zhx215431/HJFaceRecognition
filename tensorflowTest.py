
import tensorflow as tf;
import numpy as np;
import matplotlib.pyplot as plt;
sess = tf.InteractiveSession()
a = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
a = tf.reshape(a,[1,20])
print(sess.run(a))
