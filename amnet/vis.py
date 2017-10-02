import amnet
import graphviz

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

    def is_valid(self):
        """
        Checks that the symbol table has a valid inverse
        by verifying that symbols is an injection
        (no two names map to the same node)
        """
        ids = set(id(v) for v in self.symbols.values())
        return len(ids) == len(self.symbols.values())


    def __init__(self, root=None):
        self.symbols = dict()

        # does not touch the tree, only assigns to it
        if root is not None:
            self.assign_names(root)

    def prefix_names(self, prefix):
        """
        Get all existing names for a given prefix
        """
        # all names beginning with the given prefix
        return [name
                for name in self.symbols.keys()
                if name.startswith(prefix)]

    def next_unique_name(self, prefix):
        """
        Generate a unique name for a given prefix 
        """
        # if there are no existing variables return new varname
        pnames = self.prefix_names(prefix)
        if len(pnames) == 0:
            retval = prefix + '0'
            assert retval not in self.symbols, 'bad prefix'
            return retval

        # otherwise, return 1 + largest value in the names
        #pnums = [int(name[len(prefix):]) for name in pnames]
        #maxnum = max(pnums)
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

    def assign_name(self, phi):
        """
        Assigns a name to phi (and only phi), 
        provided it does not already have a name
        in the naming context
        """
        visited = (self.name_of(phi) is not None)

        if not visited:
            # get a unique name
            name = self.next_unique_name(prefix=NamingContext.default_prefix_for(phi))

            # assign the name
            self.symbols[name] = phi

        return visited

    def assign_names(self, phi):
        """
        Recursively walks the tree for phi,
        and assigns a name to each node
        """
        visited = self.assign_name(phi)

        if visited:
            #print 'Found a previously visited node (%s)' % self.name_of(phi),
            return

        if hasattr(phi, 'x'):
            self.assign_names(phi.x)
        if hasattr(phi, 'y'):
            assert isinstance(phi, amnet.Mu) or \
                   isinstance(phi, amnet.Stack)
            self.assign_names(phi.y)
        if hasattr(phi, 'z'):
            assert isinstance(phi, amnet.Mu)
            self.assign_names(phi.z)

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

    def rename(self, phi, newname):
        oldname = self.name_of(phi)
        assert oldname is not None, 'nothing to rename'
        assert newname and newname[0].isalpha(), 'invalid new name'

        # maintain invariant
        assert self.is_valid()

        # if newname already exists, rename that variable first
        if newname in self.symbols:
            phi2 = self.symbols[newname]
            name2 = self.next_unique_name(
                prefix=NamingContext.default_prefix_for(phi2)
            )
            print 'Warning (rename): %s exists within context, moving existing symbol to %s' % \
                  (newname, name2)
            self.symbols[name2] = phi2

        # now simply reattach the object
        self.symbols[newname] = self.symbols[oldname]
        del self.symbols[oldname]

        # maintain invariant
        assert self.is_valid()

    def merge_ctx(self, other_ctx):
        """
        Merges another naming context into the current context, keeping the
        same names if possible. 
        Renames the symbols from the other context when necessary.
        """
        assert self.is_valid()

        # for asserts
        nodes_premerge = len(self.symbols)
        nodes_added = 0

        for other_name, other_phi in other_ctx.symbols.items():
            here_name = self.name_of(other_phi)
            if here_name is None:
                # other node is not in current ctx,
                # try to keep the same name
                if other_name not in self.symbols:
                    self.symbols[other_name] = other_phi
                else:
                    other_name2 = self.next_unique_name(
                        prefix=NamingContext.default_prefix_for(other_phi)
                    )
                    self.symbols[other_name2] = other_phi

                nodes_added += 1
            else:
                # node already in current context, keep it
                print 'Warning (merge): %s already in destination context as %s' % \
                      (k, vname)

        # keep invariant
        nodes_postmerge = len(self.symbols)
        assert nodes_postmerge == nodes_premerge + nodes_added
        assert self.is_valid()


def _node_for(ctx, name, dot, append_dims=True):
    """
    creates a node for phi in the dot object with appropriate
    formatting
    """
    phi = ctx.symbols[name]

    if append_dims:
        label = '%s[%d->%d]' % (name, phi.indim, phi.outdim)
    else:
        label = name

    shape = None
    style = None
    fillcolor = None

    # the checking order should go *up* the class hierarchy
    if isinstance(phi, amnet.Variable):
        shape = 'box'
        fillcolor = 'gray80'
    elif isinstance(phi, amnet.Linear):
        pass
    elif isinstance(phi, amnet.Constant):
        pass
    elif isinstance(phi, amnet.Affine):
        pass
    elif isinstance(phi, amnet.Mu):
        shape = 'box'
        style = 'filled'
        fillcolor = 'gray80'
    elif isinstance(phi, amnet.Stack):
        shape = 'box'
    elif isinstance(phi, amnet.Amn):
        pass

    dot.node(name=name,
             label=label,
             shape=shape,
             style=style,
             fillcolor=fillcolor)

def amn2gv(phi, ctx=None, title=None):
    # walk the tree to find unique nodes and assign names to them
    if ctx is None:
        ctx = NamingContext(phi)

    # generate the signal graph
    sg = ctx.signal_graph()

    # create the visualization object
    dot = graphviz.Digraph()
    dot.graph_attr['rankdir'] = 'BT'
    if title:
        dot.graph_attr['labelloc'] = 't'
        dot.graph_attr['fontname'] = 'Courier New'
        dot.graph_attr['label'] = title

    # default node style
    dot.node_attr['fontname'] = 'Courier New'
    dot.node_attr['fontsize'] = '11'
    dot.node_attr['margin'] = '0.1,0.05'  # 0.11,0.055 default
    dot.node_attr['height'] = '0.3'
    dot.node_attr['color'] = 'gray50'
    dot.node_attr['shape'] = 'box'
    dot.node_attr['style'] = 'filled, rounded'
    dot.node_attr['fillcolor'] = 'white'

    # default edge style
    dot.edge_attr['color'] = 'gray60'
    dot.edge_attr['arrowsize'] = '0.6'
    dot.edge_attr['arrowhead'] = 'normal'

    for n1 in sg:
        # create node
        _node_for(ctx, n1, dot, append_dims=True)

        # create edges
        assert len(sg[n1]) <= 3

        if len(sg[n1]) < 3:
            # not a mu-node
            for n2 in sg[n1]:
                dot.edge(n2, n1)
        else:
            # mu-node
            dot.edge(sg[n1][0],   # x-input
                     n1,
                     arrowhead='dot')
            dot.edge(sg[n1][1],  # y-input
                     n1,
                     arrowhead='odot')
            dot.edge(sg[n1][2],  # z-input
                     n1,
                     arrowhead='normal')

    # return the dot object
    return dot
