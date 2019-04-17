import tensorflowWhileLoop
import tensorflow as tf
whileLoop = tensorflowWhileLoop.tensorflow_while_loop(None,None,2)
result = whileLoop.neighborhoodDifference()
shape = tf.shape(result)

sess = tf.InteractiveSession()
print(sess.run(result))
print(sess.run(shape))
