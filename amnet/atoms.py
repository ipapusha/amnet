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


def make_max2_s(phi):
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


def make_max2(phi1, phi2):
    assert phi1.outdim == 1 and \
           phi2.outdim == 1

    return make_max2_s(amnet.stack(phi1, phi2))


def _foldl(f, z, xs):
    if len(xs) == 0:
        return z
    return _foldl(f, f(z, xs[0]), xs[1:])

