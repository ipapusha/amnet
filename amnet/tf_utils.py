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

    # get topology of network
    dimensions = [weight.shape[0] for weight in weights]
    num_layers = len(dimensions)
    
    # define the input layer for the amnet
    input_vars = amnet.Variable(dimensions[0])
    prev_layer_vars = input_vars

    # compose each layer
    for n in range(num_layers):
        affine_edges = amnet.Affine(np.transpose(weights[n]), prev_layer_vars, biases[n])
        prev_layer_vars = atoms.relu(affine_edges)

    # return full construction
    return prev_layer_vars
