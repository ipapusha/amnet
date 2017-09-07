import numpy as np
import amnet
from amnet.util import foldl

import z3

def _max2_z3(x, y): return z3.If(x <= y, y, x)
def _maxN_z3(xs):
    assert len(xs) >= 1
    return foldl(_max2_z3, xs[0], xs[1:])

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
    print 'Initializing stability_search1'

    # init counterexample set
    Xc = list()
    Xc.append(np.ones(n))

    # init SMT solver
    esolver = z3.Solver()
    fsolver = z3.Solver()
    enc = amnet.smt.SmtEncoder(phi, solver=fsolver)
    enc.init_tree()

    print enc

    # 1. E-solve
    Avar = [z3.RealVector('A' + str(i), n) for i in range(m)]
    bvar = z3.RealVector('b', m)

    esolver.push()

    for k, xk in enumerate(Xc):
        # point value
        xk_next = phi.eval(xk)

        # Lyapunov max expressions
        Vk_terms = list()
        Vk_next_terms = list()
        for i in range(n):
            Vk_terms.append(
                z3.Sum(
                    [Avar[i][j] * xk[j] for j in range(n) if xk[j] != 0]
                )
            )
            Vk_next_terms.append(
                z3.Sum(
                    [Avar[i][j] * xk_next[j] for j in range(n) if xk_next[j] != 0]
                )
            )
        Vk_expr = _maxN_z3(Vk_terms)
        Vk_next_expr = _maxN_z3(Vk_next_terms)

        # Lyapunov function constraints for counterexample xk
        Vk = z3.Real('v' + str(k))
        Vk_next = z3.Real('v' + str(k) + '_next')
        esolver.add(Vk == Vk_expr)
        esolver.add(Vk_next == Vk_next_expr)

        # nonnegativity of V
        if not all(xk == 0):
            esolver.add(Vk > 0)

        # V has to decrease
        esolver.add(Vk_next < Vk)

    # find a candidate Lyapunov function
    if esolver.check():
        print 'Found new Lyapunov Function'
        m = esolver.model()
        print m
        #A_cand = [ [ m[Avar[i][j]] for j in range(n) ] for i in range(m) ]
        #b_cand = [ m[bvar[i]] for i in range(m) ]

        #print A_cand
        #print b_cand
    else:
        print 'Stability unknown'
        return None

    esolver.pop()

    # 2. F-solve

    pass
