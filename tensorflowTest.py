import PRICNN as pr
import tensorflow as tf;
import numpy as np;
import matplotlib.pyplot as plt;
sess = tf.InteractiveSession()
a = [[1,1,1],[2,2,2],[3,3,3]]
b = [[3,3,3],[2,2,2],[1,1,1]]
c = tf.subtract(a,b)
print(sess.run(c))
