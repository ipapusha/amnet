import tensorflow as tf
import numpy as np
import sys
sys.path.append('..') # so that amnet can be imported
import amnet
from amnet import tf_utils
from sklearn import preprocessing

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def main():

    # hyperparameters
    learning_rate = 0.5
    epochs = 10
    batch_size = 100
    inner_dim = 4
    output_dim = 1

    # Grab spam email data
    spam_email = []
    spam_labels = []
    # read the training data
    # spambase comes in as an ordered chunk of spam and not-spam, rearrange to get a proper dataset
    with open('spambase.csv') as f:
        for line in f:
            curr = line.split(',')
            new_curr = [1]
            for item in curr[:len(curr) - 1]:
                new_curr.append(float(item))
            spam_email.append(new_curr)
            spam_labels.append([float(curr[-1])])
    spam_email = np.array(spam_email)
    spam_email = preprocessing.scale(spam_email)  # feature scaling
    spam_labels = np.array(spam_labels)
    # Shuffle randomly to create batches
    spam_email, spam_labels = unison_shuffled_copies(spam_email, spam_labels)
    # the first 2500 out of 3000 emails will serve as training data
    train_spam_email = spam_email[0:2500]
    train_spam_labels = spam_labels[0:2500]
    # the rest 500 emails will serve as testing data
    test_spam_email = spam_email[2500:]
    test_spam_labels = spam_labels[2500:]

    reduced_dim = test_spam_email.shape[1]

    # declare the training data placeholders
    # input x - for 28 x 28 pixels = 784
    x = tf.placeholder(tf.float32, [None, reduced_dim])
    # now declare the output data placeholder - 10 digits
    y = tf.placeholder(tf.float32, [None, output_dim])

    # declare the weights connecting the input to the hidden layer
    tf.set_random_seed(0)
    w1 = tf.Variable(tf.random_normal([reduced_dim, inner_dim], stddev=0.03), name='w1')
    b1 = tf.Variable(tf.random_normal([inner_dim]), name='b1')
    # and the weights connecting the hidden layer to the output layer
    w2 = tf.Variable(tf.random_normal([inner_dim, output_dim], stddev=0.03), name='w2')
    b2 = tf.Variable(tf.random_normal([output_dim]), name='b2')

    # calculate the output of the hidden layer
    hidden_out = tf.add(tf.matmul(x, w1), b1)
    hidden_out = tf.nn.relu(hidden_out)

    # calculate the hidden layer output - in this case, we use the sigmoid for Spam/Not_Spam
    y_ = tf.nn.sigmoid(tf.add(tf.matmul(hidden_out, w2), b2))
    y_raw = tf.add(tf.matmul(hidden_out, w2), b2)
    y_sat = tf.subtract(tf.nn.relu(y_raw),tf.nn.relu(tf.add(-1.,y_raw)))

    # now let's define the cost function which we are going to train the model on
    y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                                                  + (1 - y) * tf.log(1 - y_clipped), axis=1))

    # add an optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # finally setup the initialisation operator
    init_op = tf.global_variables_initializer()

    # define an accuracy assessment operation
    correct_prediction = tf.equal(tf.round(y), tf.round(y_sat))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf_prediction = 0 # used to check correct behavior


    # start the session
    with tf.Session() as sess:
        # initialize the variables
        sess.run(init_op)
        # grab a batch
        total_batch = int(len(train_spam_labels) / batch_size)
        print("Starting training")
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                batch_x = train_spam_email[i*batch_size : (i+1)*batch_size]
                batch_y = train_spam_labels[i*batch_size : (i+1)*batch_size]
                # feed batches
                _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                avg_cost += c / total_batch
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))

        print("\nTraining complete!")
        print(sess.run(accuracy, feed_dict={x: test_spam_email, y: test_spam_labels}))

        weights, biases = tf_utils.get_vars(tf.trainable_variables(), sess)

    # create ReLU AMN
    nn = tf_utils.sat_amn(weights, biases)

    # check if the networks achieve the same classification rate 
    corrects = 0
    totals = 0
    for image, label in zip(test_spam_email, test_spam_labels):
        guess = nn.eval(image)
        guess_num = round(guess,0)
        actual_num = round(label,0)
        if guess_num == actual_num:
            corrects += 1
        totals += 1

    print('Correct classification of AMNet: ' + str(float(corrects)/float(totals)))

if __name__ == '__main__':
    main()
