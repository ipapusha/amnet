import numpy as np
from amnet import Variable, Linear
from amnet.atoms import negate, min_all
from amnet.lyap import verify_forward_invariance, VerificationResult

# define a positive discrete-time system x(t+1) = f(x(t)),
# where f(x(t)) = A*x(t), and A is entrywise positive
x = Variable(2, name='x')
A1 = np.array([[1, 2], [3, 4]])
f1 = Linear(A1, x)

# define a forward invariant set R^n_+ = {x | -min_i(x_i) <= 0)
V = negate(min_all(x))

# verify forward invariance of f
result = verify_forward_invariance(f1, V)
if result.code == VerificationResult.SUCCESS:
    print 'f1 is forward invariant'
else:
    print 'f1 is not forward invariant'
    print "Counterexample: x(t) = %s, x(t+1) = %s" % (result.x, result.xp)

# define a new system that is not forward invariant
A2 = np.array([[1, 2], [-3, 4]])
f2 = Linear(A2, x)

# verify that f is not forward invariant
result = verify_forward_invariance(f2, V)
if result.code == VerificationResult.SUCCESS:
    print 'f2 is forward invariant'
else:
    print 'f2 is not forward invariant'
    print "Counterexample: x(t) = %s, x(t+1) = %s" % (result.x, result.xp)