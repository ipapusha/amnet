from __future__ import division
import numpy as np
from amnet import Variable, Linear, Constant
from amnet import atoms, lyap

n = 5
A = np.array([[0, 1/4, 1/4, 1/4, 1/4],
              [1/4, 0, 1/4, 1/4, 1/4],
              [1/4, 1/4, 0, 1/4, 1/4],
              [1/4, 1/4, 1/4, 0, 1/4],
              [1/4, 1/4, 1/4, 1/4, 0]])
print 'A=%s' % str(A)
assert A.shape == (n, n)

x = Variable(n, name='x')

# linear consensus
# f = Linear(A, x)
# nonlinear consensus
deltas = atoms.sat(
    Linear(
        A - np.eye(n),
        x
    )
)
f = atoms.add2(x, deltas)

# verify the forward invariance of the consensus subspace
# S = {x | x_1 = ... = x_n }
V = atoms.sub2(
    atoms.max_all(x),
    atoms.min_all(x)
)
# V = atoms.norminf(
#     atoms.sub2(
#         atoms.scale(n, x),
#         Linear(
#             np.ones((n,n)),
#             x
#         )
#     )
# )
# V = atoms.norminf(
#     Linear(
#         n*np.eye(n) - np.ones((n, n)),
#         x
#     )
# )

# should be forward invariant
result = lyap.verify_forward_invariance(f, V)
if result.code == lyap.VerificationResult.SUCCESS:
    print 'f is forward invariant'
else:
    print 'f is not forward invariant'
    print "Counterexample: x(t) = %s, x(t+1) = %s" % (result.x, result.xp)
