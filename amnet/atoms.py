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
# untested methods

def make_const(b, invar):
    outdim = len(b)
    indim = invar.outdim
    return amnet.AffineTransformation(
        np.zeros((outdim, indim)),
        invar,
        b
    )


def make_relu1(x):
    """ returns max(x, 0), if x is 1-dimensional"""
    n = x.outdim
    assert n == 1
    assert isinstance(x, amnet.Variable)

    return amnet.Mu(
        x,
        make_const(np.zeros(n), x),
        make_neg(x)
    )


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
    max012 = make_max2(phi0_max12)

    return max012


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
