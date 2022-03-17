from typing import Optional
from camera_match import Node
from camera_match.optimise import optimise_pipeline

class Pipeline:
    def __init__(self, nodes: Optional[list[Node]]=None, finetune: bool=False):
        self.nodes = nodes

        if self.nodes is None:
            self.nodes = []

        self.finetune = finetune

    def solve(self, source, target):
        for node in self.nodes:
            source, target = node.solve(source, target)

        if self.finetune:
            self.nodes = optimise_pipeline(self.nodes, source, target)
            return (self.apply(source), target)

        return (source, target)

    def apply(self, RGB):
        for node in self.nodes:
            RGB = node.apply(RGB)

        return RGB
