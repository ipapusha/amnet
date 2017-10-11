import numpy as np
import amn as amnet
import atoms

# used to extract weights from tf network
# assumes that the tf variables are named 'w#' and 'b#' where # is the layer number
def get_vars(var_set, sess):
    weights = []
    biases = []
    for v in var_set:
        if v.name[0] == 'w':
            new_weight = sess.run(v)
            weights.append(new_weight)
        elif v.name[0] == 'b':
            new_bias = sess.run(v)
            biases.append(new_bias)

    # ensure that the weights and biases match dimensionality
    assert(len(weights) == len(biases))

    for i in range(len(weights)):
        if i > 0:
            assert(weights[i].shape[0] == weights[i-1].shape[1])
        assert(weights[i].shape[1] == biases[i].shape[0])

    return weights, biases


def relu_amn(weights, biases):

    # ensure that the construction makes sense
    assert(len(weights) == len(biases))

    for i in range(len(weights)):
        if i > 0:
            assert(weights[i].shape[0] == weights[i-1].shape[1])
        assert(weights[i].shape[1] == biases[i].shape[0])
    
    # define the input layer for the amnet
    input_vars = amnet.Variable(weights[0].shape[0])

    # generate network
    network = atoms.relu_net(weights, input_vars, biases)

    # return full construction
    return network
