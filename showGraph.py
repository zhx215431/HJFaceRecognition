import tensorflow as tf
data_path = 'E:/study/DL/HJFaceRecognition/project/model1.pb'


graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
graph_def.ParseFromString(tf.gfile.FastGFile(data_path,'rb').read())
tf.import_graph_def(graph_def, name='graph')
summaryWriter = tf.summary.FileWriter('log/model1',graph)
