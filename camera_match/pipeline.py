from colour import cctf_decoding, cctf_encoding

class Pipeline():
    def __init__(self, nodes):
        for node in nodes:
            if node is not None:
                if not isinstance(node, Node):
                    raise TypeError(f"{node} is not a Node.")

        self.nodes = nodes

    def solve(self, source, target):
        pass

    def apply(self, RGB):
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__} nodes={repr(self.plugins)}>"


Pipeline([
    Curves(),
    CST(),
    RootPolynomial(),
    CST(),
]).solve(source, target).process(source)

