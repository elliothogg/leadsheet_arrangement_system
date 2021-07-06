import tensorflow as tf
import numpy as np


with tf.compat.v1.Session() as sess: 
    x = tf.compat.v1.placeholder("float32", None)
    y = x**2
    result = sess.run(y, feed_dict= {x: [1,2,3]})
    print(result)