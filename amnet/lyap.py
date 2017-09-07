import numpy as np
import amnet

import z3


def stability_search1(phi, x, m):
    """ Attempts to find a max-affine Lyapunov function
        that proves global stability of an AMN:
        
        Dynamical system: x(t+1) = phi(x(t))
        Lyapunov function: V(x) = max(Ax+b),
                           A (m-by-n) and b (m)
        
        A and b are recalculated again at every E-solve step
        x should be a ref to the input variable to phi
    """
    n = phi.outdim
    assert n == x.outdim
    assert m >= 1

    # 0. Initialize
    # counterexample set
    Xc = list()
    Xc.append(np.ones(n))

    # 1. E-solve
    Avar = [z3.RealVector('A' + str(i), n) for i in range(m)]

    # 2. F-solve

    pass
