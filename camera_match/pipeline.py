from typing import Optional
from camera_match import Node

class Pipeline(Node):
    def __init__(self, nodes: Optional[list[Node]]=None):
        self.nodes = nodes

        if self.nodes is None:
            self.nodes = []

        for node in self.nodes:
            if isinstance(node, list):
                if len(node) > 2:
                    raise ValueError(
                        f"Cannot have more than two nodes for a single step of the pipeline."
                    )

    def solve(self, source, target):        
        for node in self.nodes:
            if isinstance(node, list):
                if len(node) == 2:
                    source_node, target_node = node
                    target = target_node(target)
                    source_node.solve(source, target)
                    source = source_node(source)
                elif len(node) == 1:
                    node[0].solve(source, target)
                    source = node[0](source)
            else:
                node.solve(source, target)
                source = node(source)

    def __call__(self, RGB):
        for node in self.nodes:
            if isinstance(node, list):
                if node:
                    RGB = node[0](RGB)
            else:
                RGB = node(RGB)

        return RGB
