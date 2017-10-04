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

print phimax
print phimax.eval([1, -2]) # returns: 1

# visualize
dot = amnet.vis.amn2gv(phimax, title='max2(var0)')
dot.render(filename='max.gv', directory='vis')

print dot