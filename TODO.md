# SMT encoder
- [x] Test relu
- [x] Test Constant() (through relu as well)
- [x] add code coverage
- [ ] add `find_fixed_point` operation that finds a solution to `x=phi(x)`

# Trees
- [x] Add the vgc atom
- [ ] Greedy simplification routines (across linear and stack)
- [ ] Fix issues in the tracker
- [ ] efficient `relu_aff` atom (combine relu() and Affine())
      (niklas)
- [ ] efficient `relu_nn` atom (full relu nn) (niklas)


# Examples
- [ ] rewrite `example_vgc` (in particular, remove `stack()`,
	  because there is now a `Stack` class)

# Training
- [ ] add ability to dualize an AMN
- [ ] add ability to train (small) AMNs using SMT
- [ ] implement backprop (steve)
