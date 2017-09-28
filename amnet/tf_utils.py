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
            weights.append(sess.run(v))
        elif v.name[0] == 'b':
            biases.append(sess.run(v))

    return weights, biases


def make_relu_amn(weights, biases):
    # get topology of network
    dimensions = [weight.shape[0] for weight in weights]
    num_layers = len(dimensions)
    
    # define the input layer for the amnet
    input_vars = amnet.Variable(dimensions[0])
    prev_layer_vars = input_vars

    # compose each layer
    for n in range(num_layers):
        affine_edges = amnet.AffineTransformation(np.transpose(weights[n]), prev_layer_vars, biases[n])
        prev_layer_vars = atoms.make_relu(affine_edges)

    # return full construction
    return prev_layer_vars
