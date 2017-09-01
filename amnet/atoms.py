import numpy as np
import amnet


################################################################################
# utility methods
################################################################################

def _validdims_mu(x, y, z):
    return (x.outdim == y.outdim) and \
           (z.outdim == 1)


def _validdims_gate(x, y, z1, z2):
    return (x.outdim == y.outdim) and \
           (z1.outdim == 1) and \
           (z2.outdim == 1)


################################################################################
# simple affine transformations
################################################################################

def make_neg(x):
    assert x.outdim >= 1
    return amnet.AffineTransformation(
        np.diag(-np.ones(x.outdim)),
        x,
        np.zeros(x.outdim)
    )


def make_stack(phi_list):
    assert not(isinstance(phi_list, amnet.Amn)) # has to be a list of Amns
    assert len(phi_list) >= 1
    assert isinstance(phi_list[0], amnet.Amn)

    if len(phi_list) == 1:
        return phi_list[0]
    return amnet.Stack(phi_list[0], make_stack(phi_list[1:]))


################################################################################
# Gates from Table 2
################################################################################

def make_and(x, y, z1, z2):
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


def make_or(x, y, z1, z2):
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


def make_not(x, y, z):
    assert _validdims_mu(x, y, z)
    return amnet.Mu(
        y,
        z,
        x
    )


def make_xor(x, y, z1, z2):
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

def make_le(x, y, z):
    assert _validdims_mu(x, y, z)
    return amnet.Mu(
        x,
        y,
        z
    )


def make_ge(x, y, z):
    assert _validdims_mu(x, y, z)
    return amnet.Mu(
        x,
        y,
        make_neg(z)
    )


def make_lt(x, y, z):
    assert _validdims_mu(x, y, z)
    return make_not(
        x,
        y,
        make_neg(z)
    )


def make_gt(x, y, z):
    assert _validdims_mu(x, y, z)
    return make_not(
        x,
        y,
        z
    )


def make_eq(x, y, z):
    assert _validdims_mu(x, y, z)
    return make_and(
        x,
        y,
        z,
        make_neg(z)
    )


def make_neq(x, y, z):
    assert _validdims_mu(x, y, z)
    return make_and(
        y,
        x,
        z,
        make_neg(z)
    )


################################################################################
# deprecated
################################################################################
def make_const(b, invar):
    outdim = len(b)
    indim = invar.outdim
    return amnet.AffineTransformation(
        np.zeros((outdim, indim)),
        invar,
        b
    )


################################################################################
# Methods from Table 1
################################################################################

def make_relu(phi):
    """ returns vector with ith component equal to max(phi_i, 0) """
    n = phi.outdim
    relus = [None for i in range(n)]

    # populate each component
    for i in range(n):
        a1 = amnet.AffineTransformation(
            np.eye(1, n, i),
            phi,
            np.array([0])
        )
        a2 = amnet.Constant(np.array([0]))
        a3 = amnet.AffineTransformation(
            -np.eye(1, n, i),
            phi,
            np.array([0])
        )
        #print a1.indim, a2.indim, a3.indim
        #print a1.outdim, a2.outdim, a3.outdim
        relus[i] = amnet.Mu(a1, a2, a3)
        assert relus[i].outdim == 1

    # return a stack of all components
    return make_stack(relus)


def make_max2(phi):
    assert phi.outdim == 2

    a1 = amnet.AffineTransformation(
        np.array([[1, 0]]),
        phi,
        np.array([0])
    )
    a2 = amnet.AffineTransformation(
        np.array([[0, 1]]),
        phi,
        np.array([0])
    )
    a3 = amnet.AffineTransformation(
        np.array([[-1, 1]]),
        phi,
        np.array([0])
    )

    return amnet.Mu(a1, a2, a3)


def make_max3(phi):
    assert phi.outdim == 3

    phi0 = amnet.select(phi, 0)
    phi12 = amnet.AffineTransformation(
        np.eye(2, 3, 1),
        phi,
        np.zeros(2)
    )

    max12 = make_max2(phi12)
    phi0_max12 = amnet.Stack(phi0, max12)

    return make_max2(phi0_max12)


def make_max4(phi):
    """ uses fewer mus than make_max(phi)"""
    assert phi.outdim == 4

    phi01 = amnet.AffineTransformation(
        np.eye(2, 4, 0),
        phi,
        np.zeros(2)
    )
    phi23 = amnet.AffineTransformation(
        np.eye(2, 4, 2),
        phi,
        np.zeros(2)
    )

    max01 = make_max2(phi01)
    max23 = make_max2(phi23)
    max01_max23 = amnet.Stack(max01, max23)

    return make_max2(max01_max23)


def make_max(phi):
    n = phi.outdim
    if n <= 1: return phi
    if n == 2: return make_max2(phi)

    phi_0 = amnet.select(phi, 0)
    phi_rest = amnet.AffineTransformation(
        np.eye(n-1, n, 1),
        phi,
        np.zeros(n-1)
    )

    assert phi_rest.outdim == n-1

    max_rest = make_max(phi_rest)
    return make_max2(amnet.Stack(phi_0, max_rest))


def make_triplexer(phi, a, b, c, d, e, f):
    assert phi.outdim == 1
    assert all([len(p) == 4 for p in [a, b, c, d, e, f]])

    x = [None] * 4
    y = [None] * 4
    z = [None] * 4
    w = [None] * 4

    # Layer 1 weights
    for i in range(3):
        x[i] = amnet.AffineTransformation(
            np.array(a[i]).reshape((1, 1)),
            phi,
            np.array(b[i]).reshape((1,))
        )
        y[i] = amnet.AffineTransformation(
            np.array(c[i]).reshape((1, 1)),
            phi,
            np.array(d[i]).reshape((1,))
        )
        z[i] = amnet.AffineTransformation(
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
    x[3] = amnet.AffineTransformation(
        np.array(a[3]).reshape((1, 1)),
        w[1],
        np.array(b[3]).reshape((1,))
    )
    y[3] = amnet.AffineTransformation(
        np.array(c[3]).reshape((1, 1)),
        w[2],
        np.array(d[3]).reshape((1,))
    )
    z[3] = amnet.AffineTransformation(
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