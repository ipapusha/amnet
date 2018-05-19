# AMNET: Affine Multiplexing Network Toolbox
AMNET is a Python toolbox that assists in building certain kinds of neural
networks, and formally verifying their behavior in-the-loop 
(under development).
![Maxvis](https://raw.githubusercontent.com/ipapusha/amnet/master/doc/fig/maxvis.png)

## Example usage
```python
import numpy as np
from amnet import Variable, Linear, Mu

# a two-dimensional input variable
x = Variable(2, name='x')

# choose components
a1 = Linear(np.array([[1, 0]]), x)
a2 = Linear(np.array([[0, 1]]), x)

# find difference
a3 = Linear(np.array([[-1, 1]]), x)

# if a3 <= 0, returns a1; otherwise a2
phimax = Mu(a1, a2, a3)

# equivalently, we can also write
# phimax = amnet.atoms.max_all(x)

print phimax
print phimax.eval([1, -2]) # returns: 1
```

## References
* I. Papusha, U. Topcu, S. Carr, N. Lauffer. "Affine Multiplexing Networks: System Analysis, Learning, and Computation," [arXiv:1805.00164](https://arxiv.org/abs/1805.00164) \[math.OC\], 2018.
