class Pipeline:
    def __init__(self, nodes=None):
        self.nodes = nodes

    def solve(self, source, target):
        for node in self.nodes:
            source, target = node.solve(source, target)

    def apply(self, RGB):
        for node in self.nodes:
            RGB = node.apply(RGB)

        return RGB
