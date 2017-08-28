import numpy as np
import amnet


# gates from Table 2
def make_and(x, y, z1, z2):
    assert z1.outdim == 1
    assert z2.outdim == 1
    assert x.outdim == y.outdim

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
    assert z1.outdim == 1
    assert z2.outdim == 1
    assert x.outdim == y.outdim

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
    assert z.outdim == 1
    assert x.outdim == y.outdim

    return amnet.Mu(
        y,
        z,
        x
    )


def make_xor(x, y, z1, z2):
    assert z1.outdim == 1
    assert z2.outdim == 1
    assert x.outdim == y.outdim

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
# untested methods

def constant(b, invar):
    outdim = len(b)
    indim = invar.outdim
    return amnet.AffineTransformation(
        np.zeros((outdim, indim)),
        invar,
        b
    )


def max2_s(phi):
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


def max2(phi1, phi2):
    assert phi1.outdim == 1 and \
           phi2.outdim == 1

    return max2_s(amnet.stack(phi1, phi2))


def _foldl(f, z, xs):
    if len(xs) == 0:
        return z
    return _foldl(f, f(z, xs[0]), xs[1:])

