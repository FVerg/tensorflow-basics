#

import tensorflow as tf

x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

res = tf.multiply(x1,x2)

print(res)

# The result will not be printed, that's because tensorflow implements lazy evalutation:
# results doesn't get calculated until it isn't asked for itself.

# So, we initialize a tensorflow session:

s = tf.Session()

# And then run it

print(s.run(res))

# At the end, we close it, releasing all the resources
s.close()
