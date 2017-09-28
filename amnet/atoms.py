import numpy as np
import amnet


################################################################################
# dimension and abstraction utilities
################################################################################

def select(phi, k):
    """ returns kth component of phi """
    assert (0 <= phi.outdim) and (k < phi.outdim)

    # optimization: prevent creating an affine transformation
    #               if phi is one-dimensional
    if phi.outdim == 1 and k == 0:
        return phi

    # otherwise, select the appropriate component
    return amnet.Linear(
        np.eye(1, phi.outdim, k),
        phi
    )


def to_list(phi):
    """ converts the output components of phi to a list """
    assert phi.outdim >= 1
    return [select(phi, k) for k in range(phi.outdim)]


def from_list(phi_list):
    """returns Stack(...Stack(phi_list[0], phi_list[1]), ... phi_list[-1]) """
    assert _valid_nonempty_Amn_list(phi_list)

    def stack2(x, y):
        return amnet.Stack(x, y)

    return amnet.util.foldl(stack2, phi_list[0], phi_list[1:])

# alternative implementation of from_list
# (previously make_stack(phi_list))
#
# def from_list(phi_list):
#     assert _valid_nonempty_Amn_list(phi_list)
#     if len(phi_list) == 1:
#         return phi_list[0]
#     return amnet.Stack(phi_list[0], from_list(phi_list[1:]))


def thread_over(f, *args):
    """
    Threads f over args

    Example:
        thread_over(f, x, y) gives a 
        vector z with components z_i = f(x_i, y_i)
    """
    assert len(args) >= 1
    assert amnet.util.allsame([arg.outdim for arg in args])
    n = args[0].outdim
    m = len(args)

    # references to internal components/ "transposed" version of args
    # i.e., xs[i][j] is the ith component of the jth argument
    #       xs[i] is the list of ith components to send to f
    xs = [[select(args[j], i) for j in range(m)] for i in range(n)]

    # compute each component of the output
    outputs = [f(*(xs[i])) for i in range(n)]

    # return a stack of all components
    return from_list(outputs)


################################################################################
# dimension checking methods
################################################################################

def _validdims_mu(x, y, z):
    return (x.outdim == y.outdim) and \
           (z.outdim == 1)


def _validdims_gate(x, y, z1, z2):
    return (x.outdim == y.outdim) and \
           (z1.outdim == 1) and \
           (z2.outdim == 1)


def _valid_nonempty_Amn_list(phi_list):
    return (not isinstance(phi_list, amnet.Amn)) and \
           (len(phi_list) >= 1) and \
           (isinstance(phi_list[0], amnet.Amn))


################################################################################
# unary operations
################################################################################

def identity(phi):
    """returns phi, wrapped in an identity affine transformation"""
    return amnet.Linear(
        np.eye(phi.outdim, phi.outdim),
        phi
    )


def neg(phi):
    """returns the negative of phi """
    assert phi.outdim >= 1
    # XXX: make more efficient by peeking inside Affine, Linear, Stack, and Mu
    return amnet.Linear(
        np.diag(-np.ones(phi.outdim)),
        phi
    )


################################################################################
# binary operations (Table 1)
################################################################################

def add2(x, y):
    """main n-d add method on which all add routines rely"""
    assert x.outdim == y.outdim
    n = x.outdim

    xy = amnet.Stack(x, y)

    return amnet.Linear(
        np.concatenate((np.eye(n), np.eye(n)), axis=1),
        xy
    )


def add_list(phi_list):
    """returns (...(phi_list[0] + phi_list[1]) + ...) + phi_list[len(phi_list)]"""
    assert _valid_nonempty_Amn_list(phi_list)
    return amnet.util.foldl(add2, phi_list[0], phi_list[1:])


def add_all(phi):
    """ returns the sum of all components of phi """
    assert phi.outdim >= 1
    return add_list(to_list(phi))


def sub2(x, y):
    assert x.outdim == y.outdim
    n = x.outdim

    xy = amnet.Stack(x, y)

    return amnet.Linear(
        np.concatenate((np.eye(n), -np.eye(n)), axis=1),
        xy
    )


def max2_1(x, y):
    """ main 1-d max method on which all max routines rely """
    assert x.outdim == 1 and y.outdim == 1
    return amnet.Mu(
        x,
        y,
        sub2(y, x)
    )


def max2(x, y):
    """ returns vector with ith component equal to max(x_i, y_i)"""
    assert x.outdim == y.outdim
    assert x.outdim >= 1

    return thread_over(
        max2_1,
        x,
        y
    )


def max_list(phi_list):
    """ 
    Returns vector of same size as phi_list[0]

    Example:
        if phi_list evaluates to [[1,2,3], [4,-5,6]]
        then max_list(phi_list) evaluates to [4, 2, 6]
    """
    assert _valid_nonempty_Amn_list(phi_list)
    return amnet.util.foldl(max2, phi_list[0], phi_list[1:])


def max_all(phi):
    """ 
    Returns the maximum element of phi 
    
    Example:
        if phi evaluates to [1,2,3],
        then max_all(phi) evaluates to 3
    """
    assert phi.outdim >= 1
    return max_list(to_list(phi))


def relu(phi):
    """ returns vector with ith component equal to max(phi_i, 0) """
    assert phi.outdim >= 1

    zero = amnet.Constant(phi, np.zeros(phi.outdim))
    return thread_over(
        max2_1,
        phi,
        zero
    )


def min2_1(x, y):
    """ main 1-d max method on which all min routines rely """
    assert x.outdim == 1 and y.outdim == 1
    return amnet.Mu(
        y,
        x,
        sub2(y, x)
    )


def min2(x, y):
    """ returns vector with ith component equal to max(x_i, y_i)"""
    assert x.outdim == y.outdim
    assert x.outdim >= 1

    return thread_over(
        min2_1,
        x,
        y
    )


def min_list(phi_list):
    assert _valid_nonempty_Amn_list(phi_list)
    return amnet.util.foldl(min2, phi_list[0], phi_list[1:])


def min_all(phi):
    assert phi.outdim >= 1
    return min_list(to_list(phi))


def max_aff(A, b, phi):
    """ returns an AMN that evaluates to 
        max_i(sum_j a_{ij} phi_j + b_i) """
    (m, n) = A.shape
    assert len(b) == m
    assert phi.outdim == n

    phi_aff = amnet.Affine(
        A,
        phi,
        b
    )

    return max_all(phi_aff)


def triplexer(phi, a, b, c, d, e, f):
    assert phi.outdim == 1
    assert all([len(p) == 4 for p in [a, b, c, d, e, f]])

    x = [None] * 4
    y = [None] * 4
    z = [None] * 4
    w = [None] * 4

    # Layer 1 weights
    for i in range(3):
        x[i] = amnet.Affine(
            np.array(a[i]).reshape((1, 1)),
            phi,
            np.array(b[i]).reshape((1,))
        )
        y[i] = amnet.Affine(
            np.array(c[i]).reshape((1, 1)),
            phi,
            np.array(d[i]).reshape((1,))
        )
        z[i] = amnet.Affine(
            np.array(e[i]).reshape((1, 1)),
            phi,
            np.array(f[i]).reshape((1,))
        )

    # Layer 1 nonlinearity
    for i in range(3):
        w[i] = amnet.Mu(
            x[i],
            y[i],
            z[i]
        )

    # Layer 2 weights
    x[3] = amnet.Affine(
        np.array(a[3]).reshape((1, 1)),
        w[1],
        np.array(b[3]).reshape((1,))
    )
    y[3] = amnet.Affine(
        np.array(c[3]).reshape((1, 1)),
        w[2],
        np.array(d[3]).reshape((1,))
    )
    z[3] = amnet.Affine(
        np.array(e[3]).reshape((1, 1)),
        w[0],
        np.array(f[3]).reshape((1,))
    )

    # Layer 2 nonlinearity
    w[3] = amnet.Mu(
        x[3],
        y[3],
        z[3]
    )

    return w[3]



################################################################################
# Floating-point functions (for testing)
################################################################################

def fp_mu(x, y, z):
    return x if z <= 0 else y


def fp_triplexer(inp, a, b, c, d, e, f):
    assert all([len(p) == 4 for p in [a, b, c, d, e, f]])

    x = [0] * 4
    y = [0] * 4
    z = [0] * 4
    w = [0] * 4

    # Layer 1 weights
    for i in range(3):
        x[i] = a[i] * inp + b[i]
        y[i] = c[i] * inp + d[i]
        z[i] = e[i] * inp + f[i]

    # Layer 1 nonlinearity
    for i in range(3):
        w[i] = fp_mu(x[i], y[i], z[i])

    # Layer 2 weights
    x[3] = a[3] * w[1] + b[3]
    y[3] = c[3] * w[2] + d[3]
    z[3] = e[3] * w[0] + f[3]

    # Layer 2 nonlinearity
    w[3] = fp_mu(x[3], y[3], z[3])

    return w[3]


################################################################################
# Gates from Table 2
################################################################################

def gate_and(x, y, z1, z2):
    assert _validdims_gate(x, y, z1, z2)
    return amnet.Mu(
        amnet.Mu(
            x,
            y,
            z1,
        ),
        y,
        z2
    )


def gate_or(x, y, z1, z2):
    assert _validdims_gate(x, y, z1, z2)
    return amnet.Mu(
        x,
        amnet.Mu(
            x,
            y,
            z1
        ),
        z2
    )


def gate_not(x, y, z):
    assert _validdims_mu(x, y, z)
    return amnet.Mu(
        y,
        z,
        x
    )


def gate_xor(x, y, z1, z2):
    assert _validdims_gate(x, y, z1, z2)
    return amnet.Mu(
        amnet.Mu(
            y,
            x,
            z1
        ),
        amnet.Mu(
            x,
            y,
            z1
        ),
        z2
    )


################################################################################
# Comparisons from Table 2
################################################################################

def cmp_le(x, y, z):
    assert _validdims_mu(x, y, z)
    return amnet.Mu(
        x,
        y,
        z
    )


def cmp_ge(x, y, z):
    assert _validdims_mu(x, y, z)
    return amnet.Mu(
        x,
        y,
        neg(z)
    )


def cmp_lt(x, y, z):
    assert _validdims_mu(x, y, z)
    return gate_not(
        x,
        y,
        neg(z)
    )


def cmp_gt(x, y, z):
    assert _validdims_mu(x, y, z)
    return gate_not(
        x,
        y,
        z
    )


def cmp_eq(x, y, z):
    assert _validdims_mu(x, y, z)
    return gate_and(
        x,
        y,
        z,
        neg(z)
    )


def cmp_neq(x, y, z):
    assert _validdims_mu(x, y, z)
    return gate_and(
        y,
        x,
        z,
        neg(z)
    )
