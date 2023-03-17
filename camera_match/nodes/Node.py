from abc import ABC, abstractmethod
from typing import Any
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from colour import LUT3D, LUT3x1D, write_LUT

class Node(ABC):
    @abstractmethod
    def solve(self, source: NDArray[Any], target: NDArray[Any]):
        pass

    @abstractmethod
    def __call__(self, RGB: NDArray[Any]) -> NDArray[Any]:
        pass

    def plot(self, projection='3d'):
        if (projection == '2d'):
            self.plot_2d()
        if (projection == '3d'):
            self.plot_3d()

    def plot_3d(self, size: int=12):
        RGB = np.reshape(LUT3D.linear_table(size), (-1, 3))
        RGB_A = self.__call__(RGB)

        ax = plt.axes(projection = '3d')
        ax.scatter(RGB_A[:, 0], RGB_A[:, 1], RGB_A[:, 2], c=list(map(tuple, RGB)))
        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
        plt.show()


    def plot_2d(self, size: int=100, xlim=None, ylim=None):
        xlim = [0, 1] if xlim is None else xlim
        ylim = [0, 1] if ylim is None else ylim

        RGB = np.reshape(LUT3x1D.linear_table(size), (-1, 3))
        RGB_A = self.__call__(RGB)

        fig, ax = plt.subplots()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.plot(RGB[:, 0], RGB_A[:, 0], 'r', label='R')
        ax.plot(RGB[:, 1], RGB_A[:, 1], 'g', label='G')
        ax.plot(RGB[:, 2], RGB_A[:, 2], 'b', label='B')
        ax.legend()


    def export_LUT(self, path="LUT.cube", size: int=33) -> None:
        LUT_cube = LUT3D.linear_table(size)
        RGB = np.reshape(LUT_cube, (-1, 3))
        LUT = LUT3D(
            np.reshape(self.__call__(RGB), LUT_cube.shape)
        )
        write_LUT(LUT, path)