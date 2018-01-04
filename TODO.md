# SMT encoder
- [x] Test relu
- [x] Test Constant() (through relu as well)
- [x] add code coverage
- [ ] add `find_fixed_point` operation that finds a solution to `x=phi(x)`

# Trees
- [x] Add the vgc atom
- [X] Greedy simplification routines (across linear and stack)
- [ ] Fix issues in the tracker
- [X] efficient `relu_aff` atom (combine relu() and Affine())
      (niklas)
- [ ] efficient `relu_nn` atom (full relu nn) (niklas)
- [ ] implement `compose_rewire`

# Tree Simplifications
- [ ] Replace all constants to derive from the smallest dimension
- [ ] Simplify matrix multiplies recursively if they result in a smaller
      dimension
- [ ] Remove identity operations
- [ ] Precompute constants offline
- [ ] Group select operations to refer to the same nodes
- [ ] Propagate negations and constant-multiplies

# Key features
- [ ] Move from numpy to rational arithmetic

# Examples
- [X] rewrite `example_vgc` (in particular, remove `stack()`,
	  because there is now a `Stack` class)

# Training
- [ ] add ability to dualize an AMN
- [ ] add ability to train (small) AMNs using SMT
- [ ] implement backprop (steve)
