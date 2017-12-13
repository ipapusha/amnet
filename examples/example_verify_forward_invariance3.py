import numpy as np
from amnet import Variable, Linear, Constant
from amnet import atoms, lyap

# generate a (nonlinear) system
n = 2   # up to 6 is bearable
np.random.seed(1)
A1 = np.round(3 * (2 * np.random.rand(n, n) - 1), 1)
A2 = np.round(3 * (2 * np.random.rand(n, n) - 1), 1)
print 'A1=%s' % str(A1)
print 'A2=%s' % str(A2)

# dynamics Amn
# x(t+1) = sat_2(A1*x(t)) - sat_1(A2*x(t))
x = Variable(n, name='x')
f1 = atoms.sat(Linear(A1, x), lo=-2, hi=2)
f2 = atoms.sat(Linear(A2, x), lo=-1, hi=1)
f = atoms.sub2(f1, f2)

# invariant region is L-infinity ball
# i.e., 0-sublevel set of the following function:
V = atoms.sub2(
    atoms.absval(
        atoms.sub2(
            atoms.norminf(x),
            Constant(x, np.array([1.5]))
        )
    ),
    Constant(x, np.array([0.5]))
)

# should be not forward invariant
result = lyap.verify_forward_invariance(f, V)
if result.code == lyap.VerificationResult.SUCCESS:
    print 'f is forward invariant'
else:
    print 'f is not forward invariant'
    print "Counterexample: x(t) = %s, x(t+1) = %s" % (result.x, result.xp)
