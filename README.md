## AMNET: Affine Multiplexing Network Toolbox

### Example usage
```
import numpy as np
import amnet

# input variable
x = amnet.Variable(2, name='x')

# choose components
x0 = amnet.Linear(np.array([[1, 0]]), x)
x1 = amnet.Linear(np.array([[0, 1]]), x)

# subtract x0 from x1
z = amnet.Linear(np.array([[-1, 1]]), x)

# maximum of x0 and x1
phimax = amnet.Mu(x0, x1, z)

print phimax
print phimax.eval([1, -2]) # == -2
```

### References
* Forthcoming...
