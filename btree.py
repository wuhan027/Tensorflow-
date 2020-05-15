class TNode(object):
    def __init__(self):
        value = 0
        left = None
        right = None


class BTree(object):
    def __init__(self):
        root = None


def get_deepest_path(bt):
    deep = 0
    if bt.root:
        nodes = [bt.root]
    else:
        nodes = []
    while nodes:
        deep += 1
        nodes0 = []
        for _n in nodes:
            if _n.left:
                nodes0.append(_n.left)
            if _n.right:
                nodes0.append(_n.right)
        nodes = nodes0
    return deep
