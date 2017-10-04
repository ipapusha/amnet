import tensorflow as tf
import numpy as np
import sys
sys.path.append('..') # so that amnet can be imported
import amnet
from amnet import tf_utils
from sklearn.decomposition import PCA
from tensorflow.examples.tutorials.mnist import input_data

# hyperparameters
learning_rate = 0.5
epochs = 10
batch_size = 100
reduced_dim = 40

def main():
    # grab mnist data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_labels = mnist.train.labels
    train_pca_images = []

    pca = PCA()
    pca.fit(mnist.train.images)

    rotation = pca.components_
    rotation_small = rotation[:reduced_dim]

    for image in mnist.train.images:
        small_transformed_image = np.dot(rotation_small, image)
        train_pca_images.append(small_transformed_image)

    test_labels = mnist.test.labels
    test_pca_images = []

    for image in mnist.test.images:
        small_transformed_image = np.dot(rotation_small, image)
        test_pca_images.append(small_transformed_image)


    # declare the training data placeholders
    # input x - for 28 x 28 pixels = 784
    x = tf.placeholder(tf.float32, [None, reduced_dim])
    # now declare the output data placeholder - 10 digits
    y = tf.placeholder(tf.float32, [None, 10])

    # declare the weights connecting the input to the hidden layer
    w1 = tf.Variable(tf.random_normal([reduced_dim, 20], stddev=0.03), name='w1')
    b1 = tf.Variable(tf.random_normal([20]), name='b1')
    # and the weights connecting the hidden layer to the output layer
    w2 = tf.Variable(tf.random_normal([20, 10], stddev=0.03), name='w2')
    b2 = tf.Variable(tf.random_normal([10]), name='b2')

    # calculate the output of the hidden layer
    hidden_out = tf.add(tf.matmul(x, w1), b1)
    hidden_out = tf.nn.relu(hidden_out)

    # calculate the hidden layer output - in this case, let's use a softmax activated
    # output layer
    y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, w2), b2))

    # need to save raw relu output to compare to amnet output
    y_relu = tf.nn.relu(tf.add(tf.matmul(hidden_out, w2), b2))

    # now let's define the cost function which we are going to train the model on
    y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                                                  + (1 - y) * tf.log(1 - y_clipped), axis=1))

    # add an optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # finally setup the initialization operator
    init_op = tf.global_variables_initializer()

    # define an accuracy assessment operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf_prediction = 0 # used to check correct behavior


    # start the session
    with tf.Session() as sess:
        # initialize the variables
        sess.run(init_op)
        total_batch = int(len(mnist.train.labels) / batch_size)
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                # batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
                batch_x = train_pca_images[i*batch_size : (i+1)*batch_size]
                batch_y = train_labels[i*batch_size : (i+1)*batch_size]
                _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                avg_cost += c / total_batch
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))

        print("\nTraining complete!")
        print(sess.run(accuracy, feed_dict={x: test_pca_images, y: test_labels}))

        tf_prediction = y_relu.eval(feed_dict={x: [test_pca_images[0]]}, session=sess)[0]

        weights, biass = tf_utils.get_vars(tf.trainable_variables(), sess)

    # create ReLU AMN
    nn = tf_utils.relu_amn(weights, biass)

    # check if the networks execute are the same
    for image in test_pca_images:
        amnet_prediction = nn.eval(test_pca_images[0])
        diff = abs(sum(amnet_prediction) - sum(tf_prediction))
        assert(diff < 0.0001)
    print("Tests passed")


if __name__ == '__main__':
    main()
