import random
import tensorflow as tf
import numpy as np
import sys
import z3

sys.path.append('..') # so that amnet can be imported
import amnet
from amnet import tf_utils
from amnet import smt
from tensorflow.examples.tutorials.mnist import input_data

from PIL import Image

def pca(A, dim):

    M = (A-np.mean(A.T,axis=1)).T # subtract the mean (along columns)

    # computing eigenvalues and eigenvectors of covariance matrix
    cov = np.cov(M)
    [latent,coeff] = np.linalg.eig(cov) 
    coeff = coeff.astype(float) # discard complex part that results from numerical error

    idx = np.argsort(latent) # sorting the eigenvalues
    idx = idx[::-1] # in ascending order

    # sorting eigenvectors according to the sorted eigenvalues
    coeff = coeff[:,idx]
    latent = latent[idx] # sorting eigenvalues

    p = np.size(coeff,axis=1)
    if dim < p and dim >= 0: # check if reduction makes sense
        red_coeff = coeff[:,range(dim)] # cutting dimensionality
    return red_coeff, coeff

def main():

    # hyperparameters
    learning_rate = 0.5
    epochs = 10
    batch_size = 100
    reduced_dim = 40
    hidden_layer_size = 12

    # grab mnist data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_labels = mnist.train.labels
    train_pca_images = []

    rotation_reduced, rotation_full = pca(mnist.train.images[:10000], reduced_dim)
   
    
    # convert training data to reduced dimension
    for image in mnist.train.images:
        small_transformed_image = np.dot(rotation_reduced.T, image)
        train_pca_images.append(small_transformed_image)

    # same procedure for test labels
    test_labels = mnist.test.labels
    test_pca_images = []

    for image in mnist.test.images:
        small_transformed_image = np.dot(rotation_reduced.T, image)
        test_pca_images.append(small_transformed_image)


    # declare the training data placeholders
    # input x - for 28 x 28 pixels = 784
    x = tf.placeholder(tf.float32, [None, reduced_dim])
    # now declare the output data placeholder - 10 digits
    y = tf.placeholder(tf.float32, [None, 10])

    # declare the weights connecting the input to the hidden layer
    w1 = tf.Variable(tf.random_normal([reduced_dim, hidden_layer_size], stddev=0.03), name='w1')
    b1 = tf.Variable(tf.random_normal([hidden_layer_size]), name='b1')
    # and the weights connecting the hidden layer to the output layer
    w2 = tf.Variable(tf.random_normal([hidden_layer_size, 10], stddev=0.03), name='w2')
    b2 = tf.Variable(tf.random_normal([10]), name='b2')

    # calculate the output of the hidden layer
    hidden_out = tf.add(tf.matmul(x, w1), b1)
    hidden_out = tf.nn.relu(hidden_out)

    # calculate the hidden layer output - in this case, let's use a softmax activated
    # output layer
    y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, w2), b2))

    # now let's define the cost function which we are going to train the model on
    y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                                                  + (1 - y) * tf.log(1 - y_clipped), axis=1))

    # add an optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # finally setup the initialisation operator
    init_op = tf.global_variables_initializer()

    # define an accuracy assessment operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf_prediction = 0 # used to check correct behavior


    # start the session
    with tf.Session() as sess:
        # initialize the variables
        sess.run(init_op)
        # grab a batch
        total_batch = int(len(mnist.train.labels) / batch_size)
        print("Starting training")
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                batch_x = train_pca_images[i*batch_size : (i+1)*batch_size]
                batch_y = train_labels[i*batch_size : (i+1)*batch_size]
                # feed batches
                _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                avg_cost += c / total_batch
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))

        print("\nTraining complete!")
        print(sess.run(accuracy, feed_dict={x: test_pca_images, y: test_labels}))

        weights, biases = tf_utils.get_vars(tf.trainable_variables(), sess)

    # create ReLU AMN
    nn = tf_utils.relu_amn(weights, biases)

    # grab original image to be perturbed (number 7)
    orig_image = mnist.test.images[0]
    
    # encode nn as smt
    stream = smt.SmtEncoder(phi=nn)
    y = stream.var_of(nn)
    x = stream.var_of_input()

    # create intermediate z3 variable
    x_image = z3.RealVector('x_image', 784)

    # add pca constraint (x == rotation_reduced^T * x_image)
    for i in range(len(x)):
        stream.solver.add(x[i] == sum([rotation_reduced.T[i][k] * x_image[k] for k in range(len(x_image))])) # element-wise multiplication

    # add perturbation relationship
    num_perts = 80
    perturbation = z3.RealVector('perturbation', num_perts)

    # add perturbation range constraint
    for var in perturbation:
        stream.solver.add(var < 1.0, var > -1.0)

    # take random index instead of just first n
    random_inds = random.sample(range(784), num_perts)
    j = 0
    for i in range(len(orig_image)):
        if i in random_inds:
            stream.solver.add(x_image[i] == perturbation[j] + orig_image[i])
            stream.solver.add(x_image[i] >= 0, x_image[i] <= 1)
            j += 1
        else:
            stream.solver.add(x_image[i] == orig_image[i])

    # add constraint that output should be a 5
    for var in y:
        if var == y[5]:
            continue
        stream.solver.add(y[5] > var)

    # parameters from https://stackoverflow.com/a/12516269
    z3.set_param('auto_config', False)
    z3.set_param('smt.case_split', 5)
    z3.set_param('smt.relevancy', 2)

    # call smt solver
    print(stream.solver.check())

    # grab model (will throw an error if unsat)
    m = stream.solver.model()

    # print out perturbations
    for pert in perturbation:
        print(float(m[pert].numerator_as_long())/float(m[pert].denominator_as_long()))



if __name__ == '__main__':
    main()
