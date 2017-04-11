# TensorFlow'ing

import tensorflow as tf

# two symbolic floating-point  scalers
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

# symbolic expression
add = tf.add(a, b)

# bind arbitrary values to them
sess = tf.Session()
binding = {a:1.5, b:2.6}
c = sess.run(add, feed_dict=binding)
print(c)


