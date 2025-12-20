import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter1d
from colour.algebra import table_interpolation_tetrahedral
from colour import read_LUT
from .Node import Node

from typing import Any, Tuple
from numpy.typing import NDArray


class LUT3D(Node):
    """
    3D LUT for color matching using Non-Linear Lattice Regression.

    Based on:
    Lin, Hai Ting, Zheng Lu, Seon Joo Kim, and Michael S. Brown.
    "Nonuniform lattice regression for modeling the camera imaging pipeline."
    In European Conference on Computer Vision, pp. 556-568.
    Springer Berlin Heidelberg, 2012.
    """

    def __init__(
        self, size: int = 33, smoothness: float = 0.1, identity_weight: float = 1e-6
    ):
        """
        Initialize 3D LUT with NLLR algorithm.

        Args:
            size: Number of nodes per dimension (e.g., 33 = 33³ grid)
            smoothness: Regularization for smoothness (higher = smoother)
            identity_weight: Regularization to preserve identity mapping
        """
        self.size = size
        self.m = size**3
        self.ls = smoothness
        self.lk = identity_weight
        self.levels = [np.linspace(0, 1, size) for _ in range(3)]
        self.b = None

    def _get_weights(self, points: NDArray[Any], levels: list) -> sparse.csr_matrix:
        """Fully vectorized trilinear interpolation weights."""
        N = points.shape[0]
        n = self.size
        coords, weights = [], []

        for d in range(3):
            p_d = np.clip(points[:, d], 0, 1)
            pos = np.searchsorted(levels[d], p_d) - 1
            pos = np.clip(pos, 0, n - 2)
            d0, d1 = levels[d][pos], levels[d][pos + 1]
            w1 = np.divide(
                (p_d - d0), (d1 - d0), out=np.zeros_like(p_d), where=(d1 - d0) != 0
            )
            coords.append(pos)
            weights.append(w1)

        W_data = np.zeros((N, 8))
        I_cols = np.zeros((N, 8), dtype=int)

        for i in range(8):
            bz, by, bx = (i >> 2) & 1, (i >> 1) & 1, i & 1
            w = (
                (weights[0] if bx else 1 - weights[0])
                * (weights[1] if by else 1 - weights[1])
                * (weights[2] if bz else 1 - weights[2])
            )
            idx = (coords[0] + bx) + (coords[1] + by) * n + (coords[2] + bz) * n**2
            W_data[:, i], I_cols[:, i] = w, idx

        rows = np.repeat(np.arange(N), 8)
        return sparse.csr_matrix(
            (W_data.ravel(), (rows, I_cols.ravel())), shape=(N, self.m)
        )

    def _build_smoothness_matrix(self) -> sparse.csr_matrix:
        """Vectorized 3D Laplacian for smoothness regularization."""
        n = self.size
        indices = np.arange(self.m).reshape((n, n, n), order="F")
        rows, cols, data = [], [], []

        for axis in range(3):
            if axis == 0:
                curr, neigh = indices[:-1, :, :].flatten(), indices[1:, :, :].flatten()
            elif axis == 1:
                curr, neigh = indices[:, :-1, :].flatten(), indices[:, 1:, :].flatten()
            else:
                curr, neigh = indices[:, :, :-1].flatten(), indices[:, :, 1:].flatten()

            r = np.arange(len(curr)) + (len(rows) // 2)
            rows.extend(np.repeat(r, 2))
            cols.extend(np.stack([curr, neigh], axis=1).flatten())
            data.extend(np.tile([1, -1], len(curr)))

        return sparse.csr_matrix((data, (rows, cols)), shape=(len(rows) // 2, self.m))

    def _solve_lut(
        self, source: NDArray[Any], target: NDArray[Any], levels: list
    ) -> NDArray[Any]:
        """Multi-channel vectorized solver for LUT coefficients."""
        W = self._get_weights(source, levels)
        S = self._build_smoothness_matrix()
        K = sparse.eye(self.m)
        A_stack = sparse.vstack([W, self.ls * S, self.lk * K]).tocsr()

        # Create identity target for regularization
        node_idx = np.arange(self.m)
        iz, iy, ix = np.unravel_index(
            node_idx, (self.size, self.size, self.size), order="F"
        )
        target_identity = np.stack(
            [levels[0][ix], levels[1][iy], levels[2][iz]], axis=1
        )

        y_stack = np.vstack(
            [target, np.zeros((S.shape[0], 3)), self.lk * target_identity]
        )
        return spsolve(A_stack.T @ A_stack, A_stack.T @ y_stack)

    def _warp_levels(self, errors: NDArray[Any], source: NDArray[Any]) -> list:
        """Adaptively warp grid levels based on error distribution."""
        new_levels = []
        for c in range(3):
            hist, _ = np.histogram(source[:, c], bins=100, range=(0, 1), weights=errors)
            hist_adj = 0.5 * gaussian_filter1d(hist.astype(float), 2) + 0.5 * np.mean(
                hist
            )
            cdf = np.cumsum(hist_adj)
            cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
            new_levels.append(
                np.interp(np.linspace(0, 1, self.size), cdf, np.linspace(0, 1, 100))
            )
        return new_levels

    def solve(
        self, source: NDArray[Any], target: NDArray[Any]
    ) -> Tuple[NDArray[Any], NDArray[Any]]:
        """
        Fit the 3D LUT to source and target color pairs.
        Uses adaptive grid warping for improved accuracy.
        """
        # Initial solve with uniform grid
        self.b = self._solve_lut(source, target, self.levels)

        # Predict and compute errors
        preds = self._get_weights(source, self.levels) @ self.b
        errors = np.linalg.norm(preds - target, axis=1)

        # Warp grid based on errors and re-solve
        self.levels = self._warp_levels(errors, source)
        self.b = self._solve_lut(source, target, self.levels)

        return (self(source), target)

    def __call__(self, RGB: NDArray[Any]) -> NDArray[Any]:
        """Apply the 3D LUT to RGB values."""
        if self.b is None:
            return RGB

        return self._get_weights(RGB, self.levels) @ self.b


class LUT2D(Node):
    """
    2D LUT for color matching using Non-Linear Lattice Regression on chromaticity.

    Based on:
    McElvain, Jon S., and Walter Gish.
    "Camera color correction using two-dimensional transforms."
    In Color and Imaging Conference, vol. 21, pp. 250-256.
    Society for Imaging Science and Technology, 2013.
    """

    def __init__(self, size: int = 33, smoothness: float = 0.1):
        """
        Initialize 2D LUT with NLLR algorithm.
        Operates on chromaticity coordinates (p, q) where p=R/sum, q=G/sum.

        Args:
            size: Number of nodes per dimension (e.g., 33 = 33² grid)
            smoothness: Regularization for smoothness (higher = smoother)
        """
        self.size = size
        self.ls = smoothness
        self.m = size**2
        self.grid_coords = np.linspace(0, 1, size)
        self.table = None  # Ratio table (size, size, 3)

    def _build_smoothness_matrix(self) -> sparse.csr_matrix:
        """
        Vectorized 2D Laplacian construction using Kronecker sums.
        Represents horizontal and vertical grid constraints.
        """
        n = self.size
        # 1D forward difference matrix
        main_diag = np.ones(n)
        off_diag = -np.ones(n - 1)
        D = sparse.diags([main_diag, off_diag], [0, 1], shape=(n - 1, n))

        # 2D Laplacian using Kronecker products
        I = sparse.eye(n)
        S_horizontal = sparse.kron(I, D)
        S_vertical = sparse.kron(D, I)

        return sparse.vstack([S_horizontal, S_vertical])

    def _get_weights(self, pq: NDArray[Any]) -> sparse.csr_matrix:
        """Vectorized bilinear weight construction for scattered data points."""
        N = pq.shape[0]
        n = self.size

        # Normalize and find indices
        coords = np.clip(pq, 0, 1) * (n - 1)
        idx0 = np.floor(coords).astype(int).clip(0, n - 2)
        idx1 = idx0 + 1
        frac = coords - idx0

        # Calculate 4-point weights
        w_p1 = frac[:, 0]
        w_p0 = 1.0 - w_p1
        w_q1 = frac[:, 1]
        w_q0 = 1.0 - w_q1

        # Map 2D indices to 1D lattice indices (F-order: p moves fastest)
        i00 = idx0[:, 0] + idx0[:, 1] * n
        i10 = idx1[:, 0] + idx0[:, 1] * n
        i01 = idx0[:, 0] + idx1[:, 1] * n
        i11 = idx1[:, 0] + idx1[:, 1] * n

        W_data = np.concatenate([w_p0 * w_q0, w_p1 * w_q0, w_p0 * w_q1, w_p1 * w_q1])
        I_cols = np.concatenate([i00, i10, i01, i11])
        rows = np.tile(np.arange(N), 4)

        return sparse.csr_matrix((W_data, (rows, I_cols)), shape=(N, self.m))

    def solve(
        self, source: NDArray[Any], target: NDArray[Any]
    ) -> Tuple[NDArray[Any], NDArray[Any]]:
        """
        Fit the 2D LUT by solving regularized least-squares problem.
        Maps chromaticity coordinates to output ratios.
        """
        # Linear sum and chromaticity coordinates
        sigma = np.sum(source, axis=-1, keepdims=True)
        sigma_safe = sigma + 1e-8  # Prevent division by zero
        pq = source[:, :2] / sigma_safe

        # Target ratios for the LUT
        ratios = target / sigma_safe

        # Construct system: (W.T @ W + lambda * S.T @ S) @ b = W.T @ y
        W = self._get_weights(pq)
        S = self._build_smoothness_matrix()

        A = (W.T @ W) + self.ls * (S.T @ S)
        B = W.T @ ratios

        # Solve for node values
        nodes = spsolve(A, B)
        self.table = nodes.reshape((self.size, self.size, 3), order="F")

        return (self(source), target)

    def __call__(self, RGB: NDArray[Any]) -> NDArray[Any]:
        """Apply the 2D LUT to RGB values via chromaticity lookup."""
        if self.table is None:
            return RGB

        shape_orig = RGB.shape
        RGB_flat = RGB.reshape(-1, 3)

        # Pre-process: compute chromaticity coordinates
        sigma = np.sum(RGB_flat, axis=-1, keepdims=True)
        sigma_safe = sigma + 1e-8
        pq = RGB_flat[:, :2] / sigma_safe

        # Vectorized bilinear lookup
        n = self.size
        coords = np.clip(pq, 0, 1) * (n - 1)

        idx0 = np.floor(coords).astype(int).clip(0, n - 2)
        idx1 = idx0 + 1
        frac = coords - idx0

        # Fetch surrounding nodes
        c00 = self.table[idx0[:, 0], idx0[:, 1]]
        c10 = self.table[idx1[:, 0], idx0[:, 1]]
        c01 = self.table[idx0[:, 0], idx1[:, 1]]
        c11 = self.table[idx1[:, 0], idx1[:, 1]]

        # Bilinear interpolation weights
        wp = frac[:, 0:1]
        wq = frac[:, 1:2]

        # Combine
        ratios_out = (
            c00 * (1 - wp) * (1 - wq)
            + c10 * wp * (1 - wq)
            + c01 * (1 - wp) * wq
            + c11 * wp * wq
        )

        # Post-process: rescale by original sum
        return (ratios_out * sigma).reshape(shape_orig)


class LUT(Node):
    """Load and apply custom LUT from file"""

    def __init__(self, path: str):
        """
        Load a LUT file.

        Args:
            path: Path to LUT file (supports .cube, .3dl, etc.)
        """
        self.LUT = read_LUT(path)

    def solve(
        self, source: NDArray[Any], target: NDArray[Any]
    ) -> Tuple[NDArray[Any], NDArray[Any]]:
        """Pass-through - CustomLUT doesn't learn, just applies."""
        return (self(source), target)

    def __call__(self, RGB: NDArray[Any]) -> NDArray[Any]:
        """Apply the loaded LUT to RGB values."""
        if self.LUT is None:
            return RGB

        return self.LUT.apply(RGB, interpolator=table_interpolation_tetrahedral)
