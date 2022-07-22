from abc import ABC, abstractmethod
from typing import Any
from numpy.typing import NDArray

class Node(ABC):
    @abstractmethod
    def solve(self, source: NDArray[Any], target: NDArray[Any]):
        pass

    @abstractmethod
    def __call__(self, RGB: NDArray[Any]) -> NDArray[Any]:
        pass
