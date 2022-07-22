from typing import Optional
from camera_match import Node

class Pipeline:
    def __init__(self, nodes: Optional[list[Node]]=None):
        self.nodes = nodes

        if self.nodes is None:
            self.nodes = []

        for node in self.nodes:
            if isinstance(node, list) or isinstance(node, tuple):
                if len(node) > 2:
                    raise ValueError(
                        f"Cannot have more than two nodes for a single step of the pipeline."
                    )
                if len(node) == 0:
                    raise ValueError(
                        f"Cannot have an empty list or tuple as a step of the pipeline."
                    )

    def solve(self, source, target):        
        for node in self.nodes:
            if isinstance(node, list) or isinstance(node, tuple):
                if len(node) == 2:
                    source_node, target_node = node
                    target = target_node.apply(target)
                    source_node.solve(source, target)
                    source = source_node.apply(source)
                elif len(node) == 1:
                    node[0].solve(source, target)
                    source = node[0].apply(source)
            else:
                node.solve(source, target)
                source = node.apply(source)

    def apply(self, RGB):
        for node in self.nodes:
            if isinstance(node, list) or isinstance(node, tuple):
                RGB = node[0].apply(RGB)
            else:
                RGB = node.apply(RGB)

        return RGB
