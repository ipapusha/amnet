import amnet
import graphviz

import copy


class NamingContext(object):
    @classmethod
    def prefix_for(cls, phi):
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
            return prefix + '0'

        # otherwise, return 1 + largest value in the names
        pnums = [int(name[len(prefix):]) for name in pnames]
        maxnum = max(pnums)

        return prefix + str(1 + maxnum)

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
        Assigns a name only phi (and only phi), 
        provided it does not already have a name
        in the naming context
        """
        visited = (self.name_of(phi) is not None)

        if not visited:
            # get a unique name
            name = self.next_unique_name(prefix=NamingContext.prefix_for(phi))

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

def _node_for(ctx, name, dot, append_dims=True):
    """
    creates a node for phi in the dot object with appropriate
    formatting
    """
    phi = ctx.symbols[name]

    if append_dims:
        label = ('%s[%d->%d]' % (name, phi.indim, phi.outdim))
    else:
        label = name

    shape = 'box'
    style = 'filled, rounded'
    color = 'gray50'
    fillcolor = 'white'
    #margin = '0.11,0.055' # default
    margin = '0.1,0.05'
    height = '0.3'
    #fontname='Times-Roman' # default
    fontname='Courier New'
    fontsize='11'

    # the checking order should go *up* the class hierarchy
    if isinstance(phi, amnet.Variable):
        shape = 'box'
        fillcolor = 'gray90'
    elif isinstance(phi, amnet.Linear):
        pass
    elif isinstance(phi, amnet.Constant):
        pass
    elif isinstance(phi, amnet.Affine):
        pass
    elif isinstance(phi, amnet.Mu):
        shape = 'box'
        fillcolor = 'gray90'
        style = 'filled'
    elif isinstance(phi, amnet.Stack):
        shape = 'box'
    elif isinstance(phi, amnet.Amn):
        pass

    dot.node(name=name,
             label=label,
             shape=shape,
             style=style,
             color=color,
             fillcolor=fillcolor,
             height=height,
             margin=margin,
             fontname=fontname,
             fontsize=fontsize)

def amn2gv(phi, title=None):
    # first do a deep copy of the amn to ensure we don't touch it
    phi_copy = copy.deepcopy(phi)

    # walk the copy to find unique nodes and assign names to them
    ctx = NamingContext(phi_copy)
    sg = ctx.signal_graph()

    # create the visualization object
    dot = graphviz.Digraph()
    dot.graph_attr['rankdir'] = 'BT'

    for n1 in sg:
        # create node
        _node_for(ctx, n1, dot, append_dims=True)

        # create edges
        for n2 in sg[n1]:
            dot.edge(n2,
                     n1,
                     color='gray70',
                     arrowhead='vee',
                     arrowsize='0.5')

    if title:
        dot.graph_attr['labelloc'] = 't'
        dot.graph_attr['fontname'] = 'Courier New'
        dot.graph_attr['label'] = title

    # return the dot object
    return dot
