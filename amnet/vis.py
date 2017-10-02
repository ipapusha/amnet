import amnet
from amnet.smt import NamingContext
import graphviz

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
