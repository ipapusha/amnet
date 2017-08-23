import numpy as np
import amnet


def constant(b, invar):
    outdim = len(b)
    indim = invar.outdim
    return amnet.AffineTransformation(
        np.zeros(outdim, indim),
        invar,
        b
    )


def max1(phi):
    assert phi.outdim == 1
    return phi


def max2(phi):
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


def maxn(phi):
    assert phi.outdim >= 2

    if phi.outdim == 2:
        return Max2(phi)

    return NotImplemented
