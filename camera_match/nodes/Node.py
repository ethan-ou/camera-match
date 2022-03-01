from abc import ABC, abstractmethod
from typing import Any, Tuple
from numpy.typing import NDArray

class Node(ABC):
    @abstractmethod
    def solve(self, source: NDArray[Any], target: NDArray[Any]) -> Tuple[NDArray[Any], NDArray[Any]]:
        pass

    @abstractmethod
    def apply(self, RGB: NDArray[Any]) -> NDArray[Any]:
        pass
