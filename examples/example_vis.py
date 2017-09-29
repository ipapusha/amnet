import numpy as np
import amnet
import amnet.vis

# a two-dimensional input variable
x = amnet.Variable(2, name='x')

# choose components
x0 = amnet.Linear(np.array([[1, 0]]), x)
x1 = amnet.Linear(np.array([[0, 1]]), x)

# subtract x0 from x1
z = amnet.Linear(np.array([[-1, 1]]), x)

# maximum of x0 and x1
phimax = amnet.Mu(x0, x1, z)

# or equivalently, we can instead write
phimax2 = amnet.atoms.max_all(x)

print phimax
print phimax.eval([1, -2]) # returns: 1

print phimax2
print phimax.eval([1, -2]) # returns: 1

# visualize
