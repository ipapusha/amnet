import numpy as np
from amnet import Variable, Linear, Constant
from amnet import atoms, lyap

# generate a (nonlinear) system
n = 5   # up to 6 is bearable
np.random.seed(1)
A = np.round(3 * (2 * np.random.rand(n, n) - 1), 1)

# dynamics Amn
# x(t+1) = sat(A*x(t))
x = Variable(n, name='x')
f = atoms.sat(Linear(A, x))

# invariant region is L-infinity ball
# i.e., 0-sublevel set of the following function:
V = atoms.sub2(
    atoms.norminf(x),
    Constant(x, np.array([1.0]))
)

# should be forward invariant
result = lyap.verify_forward_invariance(f, V)
if result.code == lyap.VerificationResult.SUCCESS:
    print 'f is forward invariant'
else:
    print 'f is not forward invariant'
    print "Counterexample: x(t) = %s, x(t+1) = %s" % (result.x, result.xp)
