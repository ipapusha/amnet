import numpy as np
import amnet


def Max2(phi):
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