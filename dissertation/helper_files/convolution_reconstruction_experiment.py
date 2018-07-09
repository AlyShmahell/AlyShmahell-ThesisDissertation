'''
import tensorflow as tf

d = tf.constant([[0, 1],[2,3],[4,5]], dtype=tf.float32, name='d')

k1 = tf.constant([[1,2],[3,4]], dtype=tf.float32, name='k1')
k2 = tf.constant([[1,2]], dtype=tf.float32, name='k2')


data   = tf.expand_dims(d, 2, name = "data")
kernel1 = tf.reshape(k1, [int(k1.shape[0]), 1, int(k1.shape[1])], name='kernel1')
kernel2 = tf.reshape(k2, [1, 2, 1], name='kernel2')

a  = tf.nn.conv1d(data, kernel1, 1, 'SAME')
b = tf.nn.conv1d(a, kernel2, 1, 'SAME')

with tf.Session() as sess:
    print ('data\n', sess.run(data), '\nshape:\n', sess.run(data).shape, '\n')
    print ('kernel1', sess.run(kernel1), '\nshape:\n', sess.run(kernel1).shape, '\n')
    print ('a', sess.run(a), '\nshape:\n', sess.run(a).shape, '\n')
    print ('kernel2', sess.run(kernel2), '\nshape:\n', sess.run(kernel2).shape, '\n')
    print ('b', sess.run(b), '\nshape:\n', sess.run(b).shape, '\n')
'''
import tensorflow as tf

d = tf.constant([[0, 1],[2,3],[4,5]], dtype=tf.float32, name='d')

k1 = tf.constant([[1]], dtype=tf.float32, name='k1')

data   = tf.expand_dims(d, 2, name = "data")
kernel1 = tf.reshape(k1, [int(k1.shape[0]), 1, int(k1.shape[1])], name='kernel1')

a  = tf.nn.conv1d(data, kernel1, 1, 'SAME')

with tf.Session() as sess:
    print ('data\n', sess.run(data), '\nshape:\n', sess.run(data).shape, '\n')
    print ('kernel1', sess.run(kernel1), '\nshape:\n', sess.run(kernel1).shape, '\n')
    print ('a', sess.run(a), '\nshape:\n', sess.run(a).shape, '\n')
