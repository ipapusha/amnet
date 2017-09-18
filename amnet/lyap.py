import numpy as np
import amnet
from amnet.util import foldl, r2f, mfp

import z3
import cvxpy

import itertools

_DEBUG_SMT2 = True


def _max2_z3(x, y):
    return z3.If(x <= y, y, x)

def _maxN_z3(xs):
    assert len(xs) >= 1
    return foldl(_max2_z3, xs[0], xs[1:])

def _abs_z3(x):
    return _max2_z3(x, -x)

def _normL1_z3(xs):
    assert len(xs) >= 1
    return z3.Sum([_abs_z3(x) for x in xs])

def _normLinf_z3(xs):
    assert len(xs) >= 1
    return _maxN_z3([_abs_z3(x) for x in xs])



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
    MAX_ITER = 50

    # init counterexample set
    Xc = list()
    #Xc.append(np.ones((n,)))
    # go around the Linf 1-ball
    for xcpoint in itertools.product([-1,1], repeat=n):
        Xc.append(np.array(xcpoint))

    # init SMT solver
    z3.set_param('auto_config', False)
    z3.set_param('smt.case_split', 4)
    esolver = z3.SolverFor('QF_LRA')
    fsolver = z3.SolverFor('QF_LRA')

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

        esolver.add(_maxN_z3(bvar) == 0)

        for k, xk in enumerate(Xc):
            # point value
            xk_next = phi.eval(xk)

            # Lyapunov max expressions
            Vk_terms = [z3.Sum([Avar[i][j]*xk[j] for j in range(n)]) + bvar[i] for i in range(m)]
            Vk_next_terms = [z3.Sum([Avar[i][j] * xk_next[j] for j in range(n)]) + bvar[i] for i in range(m)]
            Vk_expr = _maxN_z3(Vk_terms)
            Vk_next_expr = _maxN_z3(Vk_next_terms)

            # Lyapunov function constraints for counterexample xk
            Vk = z3.Real('v' + str(k))
            Vk_next = z3.Real('v' + str(k) + '_next')
            esolver.add(Vk == Vk_expr)
            esolver.add(Vk_next == Vk_next_expr)

            # nonnegativity/decrement of V
            if all(xk == 0):
                esolver.add(Vk == 0)
            else:
                # CONDITIONING: impose minimum decay rate
                esolver.add(Vk > 0)
                esolver.add(Vk_next < 0.99 * Vk)

            # CONDITIONING: impose upper bound on b
            esolver.add(_normL1_z3(bvar) <= 10)
            esolver.add(_maxN_z3(bvar) == 0)

            #esolver.add([bvar[i] == 0 for i in range(m)])

            # CONDITIONING: impose normalization on A
            for i in range(m):
                esolver.add(_normL1_z3(Avar[i]) <= 10)

        if _DEBUG_SMT2:
            filename = 'log/esolver_%s.smt2' % iter
            file = open(filename, 'w')
            print 'Writing %s...' % filename,
            file.write('(set-logic QF_LRA)\n')
            file.write(esolver.to_smt2())
            file.write('(get-model)')
            print 'done!'
            file.close()

        # find a candidate Lyapunov function
        if esolver.check() == z3.sat:
            print 'iter=%s: Found new Lyapunov Function' % iter
            #print 'esolver=%s' % esolver
            model = esolver.model()
            A_cand = np.array(
                [[mfp(model, Avar[i][j]) for j in range(n)] for i in range(m)]
            )
            b_cand = np.array(
                [mfp(model, bvar[i]) for i in range(m)]
            )
            print "V(x)=max(Ax+b):"
            print "A=" + str(A_cand)
            print "b=" + str(b_cand)
        else:
            print 'iter=%s: Stability unknown, exiting' % iter

            esolver.pop()
            return None

        esolver.pop()

        # 2. F-solve
        # find counterexample for candidate Lyapunov function

        fsolver.push()

        # z3 symbol for input to phi
        x = enc.get_symbol(xsys)

        # encode Vx
        Vx_terms = [z3.Sum([A_cand[i][j] * x[j] for j in range(n)]) + b_cand[i] for i in range(m)]
        Vx_expr = _maxN_z3(Vx_terms)
        Vx = z3.Real('vx')
        fsolver.add(Vx == Vx_expr)

        # z3 symbol for phi(x)
        x_next = enc.get_symbol(phi)

        # encode Vx_next
        Vx_next_terms = [z3.Sum([A_cand[i][j] * x_next[j] for j in range(n)]) + b_cand[i] for i in range(m)]
        Vx_next_expr = _maxN_z3(Vx_next_terms)
        Vx_next = z3.Real('vx_next')
        fsolver.add(Vx_next == Vx_next_expr)

        # encode failure to decrement
        fsolver.add(z3.Not(x==0))
        fsolver.add(z3.Not(z3.And(Vx > 0, Vx_next - Vx < 0)))

        # CONDITIONING: only care about small counterexamples
        fsolver.add(_normL1_z3(x) <= 5)
        fsolver.add(_normL1_z3(x) >= 0.5)

        # CONDITIONING: counterexample should not be zero
        fsolver.add([x[j] != 0 for j in range(n)])

        if _DEBUG_SMT2:
            filename = 'log/fsolver_%s.smt2' % iter
            file = open(filename, 'w')
            print 'Writing %s...' % filename,
            file.write('(set-logic QF_LRA)\n')
            file.write(fsolver.to_smt2())
            file.write('(get-model)\n')
            print 'done!'
            file.close()

        # look for a counterexample
        if fsolver.check() == z3.sat:
            print 'iter=%s: Found new Counterexample' % iter
            #print 'fsolver=%s' % fsolver
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
            return (A_cand, b_cand)


        fsolver.pop()

    # max iterations reached
    print 'Max iterations reached'
    return None

################################################################################
# local solver
################################################################################


def find_local_counterexample(phi, xsys, A, b):
    """
    Returns None if V(x) = max(Ax+b) is a local 
    Lyapunov function for the autonomous system
    x(t+1) = phi(x(t))
    
    Otherwise, returns a counter example xc
    e.g. at which !(V(phi(xc)) <= 0.99 V(xc))
    """
    (m, n) = A.shape
    assert len(b) == m
    assert phi.outdim == n
    assert xsys.outdim == n  # reference to input variable
    assert n >= 1
    assert m >= 1

    # encode the dynamics network
    fsolver = z3.Solver()
    enc = amnet.smt.SmtEncoder(phi, solver=fsolver)
    enc.init_tree()

    # references to z3 input and output variables
    xc0 = enc.get_symbol(xsys)
    xc1 = enc.get_symbol(phi)

    # encode V(xc0) and V(xc1)
    Vc0 = z3.Real('vc0')
    Vc1 = z3.Real('vc1')
    Vc0_expr = _maxN_z3(
        [z3.Sum([A[i][j] * xc0[j] for j in range(n)]) + b[i] for i in range(m)]
    )
    Vc1_expr = _maxN_z3(
        [z3.Sum([A[i][j] * xc1[j] for j in range(n)]) + b[i] for i in range(m)]
    )
    fsolver.add(Vc0 == Vc0_expr)
    fsolver.add(Vc1 == Vc1_expr)

    # V is a local Lyapunov function if, for all x in S
    # 1) V(0) == 0
    # 2) x != 0 -> V(x) > 0
    # 3) V(phi(x)) <= 0.99*V(x)
    # 4) V is radially unbounded

    # condition 1
    if np.max(b) > 0:
        print 'Not Lyapunov (zero)'
        xc = np.zeros(n)
        return xc

    # condition 4 (<-> reformulate to just on the boundary)
    #             (therefore, includes condition 2)
    fsolver.push()
    fsolver.add(_normL1_z3(xc0) == 1)
    fsolver.add(z3.Not(Vc0 > 0))

    if fsolver.check() == z3.sat:
        model = fsolver.model()
        xc  = np.array([mfp(model, xc0[j]) for j in range(n)])
        xcn = np.array([mfp(model, xc1[j]) for j in range(n)])
        print 'Not Lyapunov (radially unbound)'
        print '(xc, xn) = (%s, %s)' % (str(xc), str(xcn))
        return xc

    fsolver.pop()

    # condition 2
    # fsolver.push()
    # fsolver.add(z3.Not(xc0 == 0))
    # fsolver.add(z3.Not(Vc0 > 0))
    #
    # if fsolver.check() == z3.sat:
    #     model = fsolver.model()
    #     print 'Not Lyapunov (>0)'
    #     xc = np.array([mfp(model, xc0[j]) for j in range(n)])
    #     return xc
    #
    # fsolver.pop()

    # condition 3
    fsolver.push()
    fsolver.add(_normL1_z3(xc0) <= 1)
    fsolver.add(z3.Not(Vc1 <= 0.99*Vc0))

    if fsolver.check() == z3.sat:
        model = fsolver.model()
        xc  = np.array([mfp(model, xc0[j]) for j in range(n)])
        xcn = np.array([mfp(model, xc1[j]) for j in range(n)])
        print 'Not Lyapunov (decrement)'
        print '(xc, xn) = (%s, %s)' % (str(xc), str(xcn))
        return xc

    fsolver.pop()

    # all conditions have been met, this is a Lyapunov function
    return None


################################################################################
# convex-based solver
# incomplete
################################################################################


def get_candidate_lyapunov(Xc, Xc_next, m, n):
    Nc = len(Xc)
    assert all([len(xc) == n for xc in Xc])
    assert len(Xc_next) == Nc
    assert all([len(xc_next) == n for xc_next in Xc_next])

    A = cvxpy.Variable(m, n)
    cons = list()
    for xc, xc_next in itertools.izip(Xc, Xc_next):
        cvxpy.max_entries(A*xc) # TODO: check multiplication

    pass


def stability_search2(phi, xsys, m):
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
    MAX_ITER = 50

    # init counterexample set
    Xc = list()
    # go around the Linf 1-ball
    for xcpoint in itertools.product([-1, 1], repeat=n):
        Xc.append(np.array(xcpoint))

    pass