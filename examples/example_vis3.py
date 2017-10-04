import numpy as np
import amnet
import amnet.vis

np.random.seed(1)

# define a triplexer
x = amnet.Variable(1, name='x')
a = 3 * (2 * np.random.rand(4) - 1)
b = 3 * (2 * np.random.rand(4) - 1)
c = 3 * (2 * np.random.rand(4) - 1)
d = 3 * (2 * np.random.rand(4) - 1)
e = 3 * (2 * np.random.rand(4) - 1)
f = 3 * (2 * np.random.rand(4) - 1)
phi_tri = amnet.atoms.triplexer(x, a, b, c, d, e, f)

# visualize triplexer
dot = amnet.vis.amn2gv(phi_tri, title='phi_tri(var0)')
dot.render(filename='phi_tri.gv', directory='vis')

print dot