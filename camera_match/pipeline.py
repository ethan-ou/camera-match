from typing import Optional
from camera_match import Node

class Pipeline:
    def __init__(self, nodes: Optional[list[Node]]=None):
        self.nodes = nodes

        if self.nodes is None:
            self.nodes = []

    def solve(self, source, target):
        for node in self.nodes:
            node.solve(source, target)
            source = node.apply(source)

    def apply(self, RGB):
        for node in self.nodes:
            RGB = node.apply(RGB)

        return RGB
