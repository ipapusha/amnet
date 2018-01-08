import amnet
import z3
from itertools import izip, chain


class NamingContext(object):
    """
    NamingContext keeps track of a mapping (symbol table)
    symbols[name -> node]
    """
    @classmethod
    def default_prefix_for(cls, phi):
        """
        Returns a string that is an appropriate prefix for 
        the amn node type of phi
        """
        # the checking order should go *up* the class hierarchy
        if isinstance(phi, amnet.Variable):
            return 'var'
        elif isinstance(phi, amnet.Linear):
            return 'lin'
        elif isinstance(phi, amnet.Constant):
            return 'con'
        elif isinstance(phi, amnet.Affine):
            return 'aff'
        elif isinstance(phi, amnet.Mu):
            return 'mu'
        elif isinstance(phi, amnet.Stack):
            return 'st'
        elif isinstance(phi, amnet.Amn):
            return 'amn'
        else:
            assert False

    @classmethod
    def multiple_contexts_for(cls, phi_list):
        """
        Returns a list of NamingContext instances corresponding to the phi_list,
        making sure that each context is name-compatible with every other.

        This method is used to assign unique names to the internal nodes of multiple
        AMN instances, which allows for them to coexist without name clashing
        within the same solver instance.

        Example:
            ctx_list = NamingContext.multiple_contexts_for([phi, psi])
            => ctx_list[0] is a NamingContext for phi
            => ctx_list[1] is a NamingContext for psi
        """
        ctx_list = []
        for phi in phi_list:
            # create a NamingContext for the current phi,
            # taking into account the already-created contexts
            if len(ctx_list) == 0:
                ctx = cls(phi=phi, other_ctxs=None)      # first iteration
            else:
                ctx = cls(phi=phi, other_ctxs=ctx_list)  # subsequent iterations

            # append the created NamingContext to the retval
            ctx_list.append(ctx)

        assert len(ctx_list) == len(phi_list)
        return ctx_list

    def is_valid(self):
        """
        Checks that the symbol table has a valid inverse
        by verifying that symbols is a bijection
        (no two names map to the same node, and there are no null pointers)
        """

        # ensure surjectivity
        for _, v in self.symbols.items():
            if v is None:
                return False

        # ensure injectivity
        ids = set(id(v) for v in self.symbols.values())
        return len(ids) == len(self.symbols.values())

    def only_one_input(self):
        """
        Checks that the symbol table has only one instance
        of an input variable
        """
        vars_seen = 0
        for v in self.symbols.values():
            if isinstance(v, amnet.Variable):
                vars_seen += 1

        return vars_seen == 1

    def __init__(self, phi=None, other_ctxs=None):
        self.symbols = dict()

        # does not touch the tree, only recursively
        # assigns names to the tree rooted at phi
        if phi is not None:
            self.assign_names(phi, other_ctxs)

    def prefix_names(self, prefix):
        """
        Get all existing names for a given prefix
        within the current context
        """
        return [name
                for name in self.symbols.keys()
                if name.startswith(prefix)]

    def next_unique_name(self, prefix, other_ctxs=None):
        """
        Generate a unique name for a given prefix 
        """
        # prefix names within current context
        pnames = self.prefix_names(prefix)

        # add prefix names from other_ctxs, if they exist,
        # to ensure unique names across multiple NamingContexts
        if other_ctxs is not None:
            for other_ctx in other_ctxs:
                pnames.extend(other_ctx.prefix_names(prefix))

        # if there are no existing variables return new varname
        if len(pnames) == 0:
            retval = prefix + '0'
            assert retval not in self.symbols, 'bad prefix'
            return retval

        # otherwise, return 1 + largest value in the names
        pnums = []
        for name in pnames:
            suffix = name[len(prefix):]
            if suffix.isdigit():
                pnums.append(int(suffix))
        if len(pnums) == 0:
            retval = prefix + '0'
            assert retval not in self.symbols, 'bad prefix'
            return retval
        maxnum = max(pnums)

        retval = prefix + str(1 + maxnum)
        assert retval not in self.symbols, 'bad prefix'
        return retval

    def name_of(self, phi):
        """
        Returns the name of Amn phi, or None if it 
        does not exist in the naming context
        """
        for k, v in self.symbols.items():
            if v is phi:
                return k

        return None

    def name_of_input(self):
        """
        Returns the name of the single Variable in this context
        """
        assert self.only_one_input()

        for k, v in self.symbols.items():
            if isinstance(v, amnet.Variable):
                return k

        return None

    def assign_name(self, phi, other_ctxs=None):
        """
        Assigns a name to phi (and only phi), 
        provided it does not already have a name
        in the naming context
        """
        visited = (self.name_of(phi) is not None)

        if not visited:
            # get a unique name
            name = self.next_unique_name(
                prefix=NamingContext.default_prefix_for(phi),
                other_ctxs=other_ctxs
            )

            # assign the name
            self.symbols[name] = phi

        return visited

    def assign_names(self, phi, other_ctxs=None):
        """
        Recursively walks the tree for phi,
        and assigns a name to each node
        """
        visited = self.assign_name(phi, other_ctxs)

        if visited:
            #print 'Found a previously visited node (%s)' % self.name_of(phi),
            return

        if hasattr(phi, 'x'):
            self.assign_names(phi.x, other_ctxs)
        if hasattr(phi, 'y'):
            assert isinstance(phi, amnet.Mu) or \
                   isinstance(phi, amnet.Stack)
            self.assign_names(phi.y, other_ctxs)
        if hasattr(phi, 'z'):
            assert isinstance(phi, amnet.Mu)
            self.assign_names(phi.z, other_ctxs)

    def signal_graph(self):
        """
        Returns a dictionary with 
        keys = names of the nodes in the naming context
        values = names of the children of the nodes
        """
        sg = dict()
        for name, phi in self.symbols.items():
            if name not in sg:
                sg[name] = list()
            if hasattr(phi, 'x'):
                sg[name].append(self.name_of(phi.x))
            if hasattr(phi, 'y'):
                sg[name].append(self.name_of(phi.y))
            if hasattr(phi, 'z'):
                sg[name].append(self.name_of(phi.z))

        return sg

    def rename(self, phi, newname, other_ctxs=None):
        oldname = self.name_of(phi)
        assert oldname is not None, 'nothing to rename'
        assert newname and newname[0].isalpha(), 'invalid new name'

        # maintain invariant
        assert self.is_valid()

        # if newname already exists, rename that variable first
        if newname in self.symbols:
            phi2 = self.symbols[newname]
            name2 = self.next_unique_name(
                prefix=NamingContext.default_prefix_for(phi2),
                other_ctxs=other_ctxs
            )
            print 'Warning (rename): %s exists within context, moving existing symbol to %s' % \
                  (newname, name2)
            self.symbols[name2] = phi2

        # now simply reattach the object
        self.symbols[newname] = self.symbols[oldname]
        del self.symbols[oldname]

        # maintain invariant
        assert self.is_valid()


class SmtEncoder(object):
    @classmethod
    def multiple_encoders_for(cls, phi_list, solver=None, push_between=True):
        """
        Returns a list of SmtEncoder instances corresponding to the phi_list,
        making sure that
            1. a default solver is created, if necessary
            2. each encoder contains a reference to the same (possibly default) solver
            3. each encoder contains a reference to a unique ctx for that phi
            4. if push_between is true, solver.push() is called between each phi
        """
        # initialize a common SMT solver, if needed
        if solver is None:
            solver = z3.Solver()

        # encode from multi context
        ctx_list = NamingContext.multiple_contexts_for(phi_list)
        enc_list = []
        for ctx in ctx_list:
            enc = cls(ctx=ctx, solver=solver)
            enc_list.append(enc)
            if push_between:
                solver.push()

        # all encoders contain a reference to the same solver
        assert len(phi_list) == len(enc_list)
        return enc_list

    @classmethod
    def multiple_encode(cls, *args):
        """
        Convenience method for default from-AMN encoding of multiple AMN references
        Example:
             enc1, enc2, ... = SmtEncoder.multiple_encode(phi1, phi2, ...)
        """
        # TODO: add functionality that allows incrementally adding AMN nodes.
        assert len(args) >= 2, 'multiple_encode(*args) called unnecessarily'
        enc_list = cls.multiple_encoders_for(list(args))
        return enc_list

    def __init__(self, phi=None, ctx=None, solver=None):
        # initialize new SMT solver if needed
        if solver is None:
            solver = z3.Solver()
        self.solver = solver

        # init from context
        if ctx is not None:
            self.encode_from_ctx(ctx)  # use existing context
        elif phi is not None:
            self.encode_from_amn(phi)  # create new default context
        else:
            raise Exception('Either ctx or phi must be provided')

    def encode_from_ctx(self, ctx):
        """
        Encodes the AMN using the specified context
        This is the preferred way to init, although a bit verbose
        """
        self.ctx = ctx
        self.vars = dict()  # name -> RealVector
        self._init_vars()   # initialize z3 vars
        assert self.ctx.is_valid()
        assert self.ctx.only_one_input()

        self._encode()      # do the encoding work

    def encode_from_amn(self, phi):
        """
        Creates a default naming context, and encodes the AMN
        """
        defaultctx = NamingContext(phi)
        self.encode_from_ctx(ctx=defaultctx)

    def var_of(self, phi):
        """
        Returns z3 variable (as a list) associated with the output of phi 
        """
        name = self.ctx.name_of(phi)
        return self.vars[name]

    def var_of_input(self):
        """
        Returns z3 variable (as a list) associated with the input
        to the embedded context
        """
        name = self.ctx.name_of_input()
        return self.vars[name]

    def _init_vars(self):
        assert self.ctx.is_valid()
        assert self.ctx.only_one_input()

        for name, phi in self.ctx.symbols.items():
            self.vars[name] = z3.RealVector(
                prefix=name,
                sz=phi.outdim
            )

    def _link_affine(self, phi, include_bterm=True):
        assert isinstance(phi, amnet.Affine) or \
               isinstance(phi, amnet.Linear)

        m, n = phi.w.shape
        assert m >= 1 and n >= 1

        # extract children
        yvar = self.var_of(phi)
        xvar = self.var_of(phi.x)
        assert (len(yvar) == m) and (len(xvar) == n)

        # go row-by-row of w
        for i in range(m):
            rowi = phi.w[i, :]
            assert len(rowi) == n

            rowsum = z3.Sum([wij * xj
                             for wij, xj in izip(rowi, xvar)
                             if wij != 0])

            if include_bterm:
                assert isinstance(phi, amnet.Affine), \
                    'Warning: tried to encode bterm in non-Affine'
                self.solver.add(yvar[i] == (rowsum + phi.b[i]))
            else:
                self.solver.add(yvar[i] == rowsum)

    def _link_mu(self, phi):
        assert isinstance(phi, amnet.Mu)

        # check dimensions
        wvar = self.var_of(phi)
        xvar = self.var_of(phi.x)
        yvar = self.var_of(phi.y)
        zvar = self.var_of(phi.z)
        assert len(xvar) == len(yvar) and \
               len(zvar) == 1
        assert len(wvar) == len(xvar)

        # go by-element of w
        for i in range(len(wvar)):
            self.solver.add(
                wvar[i] == z3.If(zvar[0] <= 0, xvar[i], yvar[i])
            )

    def _link_stack(self, phi):
        assert isinstance(phi, amnet.Stack)

        # check dimensions
        wvar = self.var_of(phi)
        xvar = self.var_of(phi.x)
        yvar = self.var_of(phi.y)
        assert len(wvar) == len(xvar) + len(yvar)

        # go by-element of w
        for left, right in izip(wvar, chain(xvar, yvar)):
            self.solver.add(left == right)

    def _link_constant(self, phi):
        assert isinstance(phi, amnet.Constant)

        # # *overwrite* the output variable of phi to be a z3 RealVal
        # assert self.ctx.is_valid()
        # name = self.ctx.name_of(phi)
        #
        # assert len(phi.b) == len(self.vars[name])  # pre-overwrite
        # self.vars[name] = [z3.RealVal(bi) for bi in phi.b] # BUG?
        # assert self.ctx.is_valid()  # post-overwrite

        assert self.ctx.is_valid()
        cvar = self.var_of(phi)

        # check dimensions
        assert len(cvar) == len(phi.b)

        # go by-element of cvar
        for ci, bi in izip(cvar, phi.b):
            self.solver.add(ci == bi)

    def _encode(self):
        """
        encodes the relationship between the nodes
        by iterating through the context
        """
        for name, phi in self.ctx.symbols.items():
            # the checking order should go *up* the class hierarchy
            if isinstance(phi, amnet.Variable):
                pass # nothing to do
            elif isinstance(phi, amnet.Linear):
                self._link_affine(phi, include_bterm=False)
            elif isinstance(phi, amnet.Constant):
                self._link_constant(phi)
            elif isinstance(phi, amnet.Affine):
                self._link_affine(phi, include_bterm=True)
            elif isinstance(phi, amnet.Mu):
                self._link_mu(phi)
            elif isinstance(phi, amnet.Stack):
                self._link_stack(phi)
            elif isinstance(phi, amnet.Amn):
                assert False, 'Failure: do not know how to encode an Amn'
            else:
                assert False
