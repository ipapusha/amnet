import tensorflow as tf
import numpy as np
import amn as amnet
import atoms
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

reduced_dim = 50

# grab mnist data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
images = mnist.test.images
image = mnist.test.images[0]

pca = PCA()
pca.fit(images)

rotation = pca.components_
rotation_small = rotation[:reduced_dim]

transformed_image = np.dot(rotation, image)
small_transformed_image = np.dot(rotation_small, image)

print(transformed_image)
print(small_transformed_image)



# plt.plot(pca.explained_variance_ratio_)
# plt.show()
