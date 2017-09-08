import numpy as np
import amnet
from amnet.util import foldl, r2f, mfp

import z3

def _max2_z3(x, y): return z3.If(x <= y, y, x)
def _maxN_z3(xs):
    assert len(xs) >= 1
    return foldl(_max2_z3, xs[0], xs[1:])


def stability_search1(phi, xsys, m):
    """ Attempts to find a max-affine Lyapunov function
        that proves global stability of an AMN:
        
        Dynamical system: x(t+1) = phi(x(t))
        Lyapunov function: V(x) = max(Ax+b),
                           A (m-by-n) and b (m)
        
        A and b are recalculated again at every E-solve step
        x should be a ref to the input variable to phi
    """
    n = phi.outdim
    assert n == xsys.outdim
    assert m >= 1

    # 0. Initialize
    print 'Initializing stability_search1'
    MAX_ITER = 10

    # init counterexample set
    Xc = list()
    Xc.append(np.zeros(n))

    # init SMT solver
    esolver = z3.Solver()
    fsolver = z3.Solver()
    enc = amnet.smt.SmtEncoder(phi, solver=fsolver)
    enc.init_tree()

    print enc

    for iter in range(MAX_ITER):

        # 1. E-solve
        esolver.push()

        Avar = [z3.RealVector('A' + str(i), n) for i in range(m)]
        bvar = z3.RealVector('b', m)

        print 'iter=%s: Xc=%s' % (iter, Xc)
        print 'Avar=%s' % Avar
        print 'bvar=%s' % bvar

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

            # nonnegativity/decrement of V
            if not all(xk == 0):
                esolver.add(Vk > 0)
                esolver.add(Vk_next - Vk < 0)
            else:
                esolver.add(Vk == 0)

        # find a candidate Lyapunov function
        if esolver.check():
            print 'iter=%s: Found new Lyapunov Function' % iter
            print 'esolver=%s' % esolver
            model = esolver.model()
            A_cand = np.array(
                [[mfp(model, Avar[i][j]) for j in range(n)] for i in range(m)]
            )
            b_cand = np.array(
                [mfp(model, bvar[j]) for j in range(n)]
            )
            print "V(x)=max(Ax+b):"
            print "A=" + str(A_cand)
            print "b=" + str(b_cand)
        else:
            print 'iter=%s: Stability unknown, exiting' % iter
            return None

        esolver.pop()

        # 2. F-solve
        # find counterexample for candidate Lyapunov function

        fsolver.push()

        # z3 symbol for input to phi
        x = enc.get_symbol(xsys)

        # encode Vx
        Vx_terms = [z3.Sum([A_cand[i][j] * x[j] for j in range(n) if A_cand[i][j] != 0]) for i in range(m)]
        Vx_expr = _maxN_z3(Vx_terms)
        Vx = z3.Real('vx')
        fsolver.add(Vx == Vx_expr)

        # z3 symbol for phi(x)
        x_next = enc.get_symbol(phi)

        # encode Vx_next
        Vx_next_terms = [z3.Sum([A_cand[i][j] * x_next[j] for j in range(n) if A_cand[i][j] != 0]) for i in range(m)]
        Vx_next_expr = _maxN_z3(Vx_next_terms)
        Vx_next = z3.Real('vx_next')
        fsolver.add(Vx_next == Vx_next_expr)

        # encode failure to decrement
        fsolver.add(z3.Not(x==0))
        fsolver.add(z3.Not(z3.And(Vx > 0, Vx_next - Vx < 0)))

        # look for a counterexample
        if fsolver.check():
            print 'iter=%s: Found new Counterexample' % iter
            print 'fsolver=%s' % fsolver
            fmodel = fsolver.model()
            xc = np.array([mfp(fmodel, x[j]) for j in range(n)])
            Xc.append(xc)
        else:
            print 'iter=%s: No Counterexample found' % iter
            print 'Lyapunov function found'

            print "V(x)=max(Ax+b):"
            print "A=" + str(A_cand)
            print "b=" + str(b_cand)

            fsolver.pop()
            return (A, b)


        fsolver.pop()