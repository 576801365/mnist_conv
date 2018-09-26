import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder('float',[None,784])
y_ = tf.placeholder('float',[None,10])

#权重初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
#卷积和池化
def conv_2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#第一层卷积
W1_conv = weight_variable([3,3,1,32])
b1_conv = bias_variable([32])
x_image = tf.reshape(x,[-1,28,28,1])

h1_conv = tf.nn.relu(conv_2d(x_image,W1_conv) + b1_conv)
h1_pool = max_pool_2(h1_conv)
#第二层卷积
W2_conv = weight_variable([3,3,32,64])
b2_conv = bias_variable([64])

h2_conv = tf.nn.relu(conv_2d(h1_pool,W2_conv) + b2_conv)
h2_pool = max_pool_2(h2_conv)
#全连接层
h2_pool_flat = tf.reshape(h2_pool,[-1,7*7*64])
W1_fc = weight_variable([7*7*64,1024])
b1_fc = bias_variable([1024])

h1_fc = tf.nn.relu(tf.matmul(h2_pool_flat,W1_fc) + b1_fc)
#Dropout
keep_prob = tf.placeholder('float')
h1_fc_drop = tf.nn.dropout(h1_fc,keep_prob)
#输出层
W2_fc = weight_variable([1024,10])
b2_fc = bias_variable([10])

y = tf.nn.softmax(tf.matmul(h1_fc_drop,W2_fc) + b2_fc)
#损失函数
loss = - tf.reduce_sum(y_*tf.log(y))
#训练过程
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(50)
        _,train_accuracy = sess.run([train_step,accuracy], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
        if i % 1 == 0:
            print('百分之', i/10,'正确率',train_accuracy)

    print('正确率 = ', sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

