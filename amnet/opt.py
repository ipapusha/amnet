from __future__ import division

import numbers

import numpy as np
import amnet
import z3
from enum import Enum
import itertools

################################################################################
# Classes and routines related to Multiplexing Programming with AMNs
################################################################################


############
# Objective
############

class Objective(object):
    """
    Class that keeps track of:
        flag to minimize or maximize
        real-valued Amn instance (ensures 1-dimensional output)
        unique variable input
    """
    def __init__(self, phi, minimize=True):
        assert phi.outdim == 1
        self.minimize = minimize
        self.phi = phi
        self.variable = amnet.tree.unique_leaf_of(phi)
        self.is_negated = False

    def negate(self):
        assert self.phi.outdim == 1
        self.phi = amnet.atoms.negate(self.phi)
        self.is_negated = (not self.is_negated)
        self.minimize = (not self.minimize)

    def __repr__(self):
        return "Objective(phi=%s, minimize=%s)" % (repr(self.phi), repr(self.minimize))


class Minimize(Objective):
    def __init__(self, phi):
        super(Minimize, self).__init__(phi, minimize=True)


class Maximize(Objective):
    def __init__(self, phi):
        super(Maximize, self).__init__(phi, minimize=False)


##############
# Constraints
##############

class Relation(Enum):
    LT = 1  # <
    LE = 2  # <=
    GT = 3  # >
    GE = 4  # >=
    EQ = 5  # ==
    NEQ = 6  # !=


class Constraint(object):
    """
    Class that keeps track of the lhs, rhs, the type of relation,
    and ensures dimensionality coherence between the lhs and rhs
    """
    def __init__(self, lhs, rhs, rel):
        # supported relations
        assert rel in [Relation.LT, Relation.LE, Relation.GT, Relation.GE, Relation.EQ, Relation.NEQ]
        self.rel = rel

        # lhs must be an Amn
        assert isinstance(lhs, amnet.Amn)
        self.lhs = lhs

        # up-convert rhs if necessary
        if isinstance(rhs, numbers.Real):
            c = Constant(
                self,
                np.repeat(rhs, self.outdim)
            )
        elif isinstance(rhs, np.ndarray):
            self.rhs = rhs
        elif isinstance(rhs, amnet.Amn):
            assert rhs.outdim == lhs.outdim
            pass
        else:
            raise TypeError("Invalid call of Constraint(lhs=%s, rhs=%s, rel=%s)" %
                            (str(lhs), str(rhs), str(rel)))

        # at this point both self.lhs and self.rhs are valid Amn's
        assert self.lhs.outdim == self.rhs.outdim
        assert self.lhs.outdim >= 1

        # cache input variable reference
        # XXX: possibly move this check into the problem creation routines
        lhs_variable = amnet.tree.unique_leaf_of(self.lhs)
        rhs_variable = amnet.tree.unique_leaf_of(self.rhs)
        assert lhs_variable is rhs_variable, 'LHS and RHS must depend on the same Variable'
        self.variable = lhs_variable

    def __repr__(self):
        return "Constraint(lhs=%s, rhs=%s, rel=%s)" % (repr(self.lhs), repr(self.rhs), repr(self.rel))


##########
# Problem
##########

class OptResultCode(Enum):
    SUCCESS = 1
    FAILURE = 2
    INFEASIBLE = 3
    UNBOUNDED_BELOW = 4  # not yet implemented
    UNBOUNDED_ABOVE = 5  # not yet implemented


class OptResult(object):
    def __init__(self, optval, optpoint, code=OptResultCode.SUCCESS, model=None):
        self.optval = optval
        self.optpoint = optpoint
        self.code = code
        self.model = model

    def __repr__(self):
        return "OptResult(optval=%s, optpoint=%s, code=%s, model=%s)" % (repr(self.optval), repr(self.optpoint), repr(self.code), repr(self.model))


class OptOptions(object):
    def __init__(self):
        # default options
        self.obj_lo = -float(2**20)
        #self.obj_lo = float(0)
        self.obj_hi = float(2**20)
        self.fptol = float(2**-1)


class Problem(object):
    # Objective (or constant)
    # list of Constraints
    # solve()
    # single variable
    def __init__(self, objective, constraints=None, options=None, solver=None):
        assert objective is not None
        self.objective = objective
        self.constraints = [] if constraints is None else constraints
        self.options = OptOptions() if options is None else options  # default options

        # default objective is zero for a feasibility problem
        if objective is None:
            assert len(constraints) >= 1
            self.variable = constraints[0].variable
            self.objective = amnet.atoms.zero_from(self.variable, dim=1)
        else:
            self.variable = self.objective.variable

        # ensure the leaf variable is the same across constraints
        assert all([constraint.variable is self.variable
                    for constraint in self.constraints])

        # initialize default solver if necessary
        if solver is None:
            self.solver = z3.Solver()
        else:
            self.solver = solver
        self.enc_list = []

        # initialize objective and constraint relations into the solver
        self._init_objective_constraints()
        self._encode_constraint_relations()

    def eval_feasible(self, xinp):
        """ returns whether or not xinp satisfies the constraints """
        assert len(xinp) == self.objective.phi.indim
        feas = True
        for constraint in self.constraints:
            lhsval = constraint.lhs.eval(xinp)
            rhsval = constraint.rhs.eval(xinp)
            rel = constraint.rel
            # TODO: implement fptol here
            if rel == Relation.LT:
                feases = [l < r for (l, r) in itertools.izip(lhsval, rhsval)]
            elif rel == Relation.LE:
                feases = [l <= r for (l, r) in itertools.izip(lhsval, rhsval)]
            elif rel == Relation.GT:
                feases = [l > r for (l, r) in itertools.izip(lhsval, rhsval)]
            elif rel == Relation.GE:
                feases = [l >= r for (l, r) in itertools.izip(lhsval, rhsval)]
            elif rel == Relation.EQ:
                feases = [l == r for (l, r) in itertools.izip(lhsval, rhsval)]
            elif rel == Relation.NEQ:
                feases = [l != r for (l, r) in itertools.izip(lhsval, rhsval)]
            else:
                assert False, 'Impossible relation'
            assert len(lhsval) == len(feases)
            feas = feas and all(feases)

            # short-circuit
            if not feas: return feas

        return feas


    def _init_objective_constraints(self):
        assert self.solver is not None
        assert self.objective.phi.outdim == 1

        # Amn instances for this problem are encoded in a particular order
        amn_list = list(itertools.chain(
            [self.objective.phi],
            [constraint.lhs for constraint in self.constraints],
            [constraint.rhs for constraint in self.constraints]
        ))
        assert len(amn_list) >= 1
        assert (not (len(self.constraints) == 0)) or (len(amn_list) == 1)
        assert (not (len(self.constraints) > 0)) or (len(amn_list) > 1)

        # encode them into a single Smt encoder
        enc_list = amnet.smt.SmtEncoder.multiple_encoders_for(
            phi_list=amn_list,
            solver=self.solver,
            push_between=False
        )
        assert len(enc_list) == len(amn_list)
        assert len(enc_list) >= 1
        assert all([enc.solver is self.solver for enc in enc_list])

        # set the encoder lists after encoding from Smt encoder
        self.enc_list = enc_list
        self.enc_objective = enc_list[0]
        assert (len(enc_list) - 1) % 2 == 0
        ncons = (len(enc_list) - 1) // 2
        self.enc_lhs_list = enc_list[1:1+ncons]
        self.enc_rhs_list = enc_list[1+ncons:1+2*ncons]
        assert len(self.enc_lhs_list) == len(self.constraints)
        assert len(self.enc_rhs_list) == len(self.constraints)
        assert 1 + len(self.enc_lhs_list) + len(self.enc_rhs_list) == len(self.enc_list)

    def _encode_constraint_relations(self):
        assert self.solver is not None
        assert len(self.enc_list) >= 1

        # input variable amn reference
        x = self.enc_objective.var_of_input()

        # Amn instances for this problem are encoded in a particular order
        for i in xrange(len(self.constraints)):
            phi_lhs = self.constraints[i].lhs
            phi_rhs = self.constraints[i].rhs
            enc_lhs = self.enc_lhs_list[i]
            enc_rhs = self.enc_rhs_list[i]
            rel = self.constraints[i].rel

            # determine z3 variables
            v_lhs = enc_lhs.var_of(phi_lhs)
            v_rhs = enc_rhs.var_of(phi_rhs)
            inp_lhs = enc_lhs.var_of_input()
            inp_rhs = enc_rhs.var_of_input()
            assert len(v_lhs) == len(v_rhs)
            assert len(v_lhs) == phi_lhs.outdim
            assert len(v_lhs) == phi_rhs.outdim
            assert len(x) == len(inp_lhs) == len(inp_rhs)

            # encode the relation
            amnet.util.relv_z3(self.solver, v_lhs, v_rhs, rel)

            # equate all the inputs of all the constraints to the input of the objective
            amnet.util.eqv_z3(self.solver, x, inp_lhs)
            amnet.util.eqv_z3(self.solver, x, inp_rhs)


    def _encode_objective_relation(self, gamma, rel=Relation.LE):
        assert self.solver is not None
        assert len(self.enc_list) >= 1

        # determine z3 variables to compare
        v_obj = self.enc_objective.var_of(self.objective.phi)
        v_obj_rhs = np.array([gamma])
        assert len(v_obj) == 1

        # encode the relation
        amnet.util.relv_z3(self.solver, v_obj, v_obj_rhs, rel)

    def _bisection_minimize(self, check_constraints=True):
        assert self.objective.minimize

        # initialize lo and hi
        lo = self.options.obj_lo
        hi = self.options.obj_hi
        assert lo <= hi

        # initialize z3 encoding of objective and constraints
        # self._init_objective_constraints()
        # self._encode_constraint_relations()

        result = OptResult(
            optpoint=None,
            optval=None,
            code=OptResultCode.FAILURE,
            model=None
        )

        if check_constraints:
            result_z3 = self.solver.check()
            if result_z3 == z3.unsat:
                result.code = OptResultCode.INFEASIBLE
                return result

        firstrun = True

        while True:
            assert lo <= hi

            if (hi - lo) <= self.options.fptol:
                # done with bisection
                # TODO: implement unboundedness check
                return result
            else:
                # reset solver state
                if not firstrun:
                    self.solver.pop()

            # try an objective value
            firstrun = False
            assert hi - lo > self.options.fptol
            self.solver.push()
            gamma = (lo + hi) / 2
            self._encode_objective_relation(gamma, rel=Relation.LE)

            # bisect
            print 'OPT: trying gamma=%d' % (gamma)
            #print 'SOLVER: %s' % self.solver
            result_z3 = self.solver.check()
            if result_z3 == z3.sat:
                print "sat"

                model = self.solver.model()

                #print 'MODEL: %s' % model

                xv = self.enc_objective.var_of_input()
                f = self.enc_objective.var_of(self.objective.phi)

                assert len(xv) == self.variable.outdim
                assert len(f) == 1

                result.optpoint = amnet.util.mfpv(model, xv)
                result.optval = amnet.util.mfp(model, f[0])
                result.code = OptResultCode.SUCCESS
                result.model = model

                # numerical feasibility check
                #print 'POINT: %s' % result.optpoint
                assert self.eval_feasible(result.optpoint)

                #print result

                # update bound
                hi = gamma

            elif result_z3 == z3.unsat:
                print "unsat"

                # update bound
                lo = gamma
            else:
                assert False, 'Invalid result from z3'

    def _bisection_maximize(self):
        # XXX: default implementation: negate the objective and minimize instead
        assert (not self.objective.minimize)
        self.objective.negate()

        assert self.objective.minimize
        return self._bisection_minimize()

    def solve(self):
        if self.objective.minimize:
            return self._bisection_minimize()
        else:
            return self._bisection_maximize()