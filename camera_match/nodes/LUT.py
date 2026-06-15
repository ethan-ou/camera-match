import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, lsqr
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
from colour.algebra import table_interpolation_tetrahedral
from colour import LUT3D as ColourLUT3D, read_LUT
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

        points_clipped = np.clip(points, 0, 1)
        coords = np.empty((N, 3), dtype=int)
        weights = np.empty((N, 3))

        for d in range(3):
            p_d = points_clipped[:, d]
            pos = np.searchsorted(levels[d], p_d) - 1
            pos = np.clip(pos, 0, n - 2)
            coords[:, d] = pos
            d0 = levels[d][pos]
            d1 = levels[d][pos + 1]
            diff = d1 - d0
            weights[:, d] = np.where(diff != 0, (p_d - d0) / diff, 0)

        # Pre-compute weight complements
        w_comp = 1 - weights

        # Binary pattern: bx, by, bz for 8 corners
        bx = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        by = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        bz = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        # Compute all weights at once using broadcasting
        wx = np.where(bx[:, None], weights[:, 0], w_comp[:, 0])  # (8, N)
        wy = np.where(by[:, None], weights[:, 1], w_comp[:, 1])
        wz = np.where(bz[:, None], weights[:, 2], w_comp[:, 2])
        W_data = (wx * wy * wz).T  # (N, 8)

        # Compute all indices at once
        I_cols = (
            (coords[:, None, 0] + bx)
            + (coords[:, None, 1] + by) * n
            + (coords[:, None, 2] + bz) * n * n
        )  # (N, 8)

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
        frac = coords - idx0

        # Pre-compute weight complements for efficiency
        w_p = frac[:, 0]
        w_q = frac[:, 1]
        w_p_comp = 1.0 - w_p
        w_q_comp = 1.0 - w_q

        # Compute all 4 weights in vectorized form (avoiding redundant multiplications)
        W_data = np.stack(
            [
                w_p_comp * w_q_comp,  # (0, 0)
                w_p * w_q_comp,  # (1, 0)
                w_p_comp * w_q,  # (0, 1)
                w_p * w_q,  # (1, 1)
            ],
            axis=1,
        ).ravel()  # (N, 4) -> (N*4,)

        # Map 2D indices to 1D lattice indices (F-order: p moves fastest)
        idx1 = idx0 + 1
        I_cols = np.stack(
            [
                idx0[:, 0] + idx0[:, 1] * n,
                idx1[:, 0] + idx0[:, 1] * n,
                idx0[:, 0] + idx1[:, 1] * n,
                idx1[:, 0] + idx1[:, 1] * n,
            ],
            axis=1,
        ).ravel()  # (N, 4) -> (N*4,)

        rows = np.repeat(np.arange(N), 4)

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

        # Pre-compute weight complements
        wp = frac[:, 0:1]
        wq = frac[:, 1:2]
        wp_comp = 1 - wp
        wq_comp = 1 - wq

        # Fetch surrounding nodes - use advanced indexing once per corner
        idx0_p, idx0_q = idx0[:, 0], idx0[:, 1]
        idx1_p, idx1_q = idx1[:, 0], idx1[:, 1]

        # Bilinear interpolation with pre-computed complements
        ratios_out = (
            self.table[idx0_p, idx0_q] * wp_comp * wq_comp
            + self.table[idx1_p, idx0_q] * wp * wq_comp
            + self.table[idx0_p, idx1_q] * wp_comp * wq
            + self.table[idx1_p, idx1_q] * wp * wq
        )

        # Post-process: rescale by original sum
        return (ratios_out * sigma).reshape(shape_orig)


# --- Hierarchical RBF basis (pure numpy/scipy) ---------------------------------
#
# This is a from-scratch reimplementation of ALGLIB's ``rbfsetalgohierarchical``
# (the V2 model), so the package no longer depends on the xalglib binary wheel.
# The algorithm and its constants were reverse-engineered by probing the library
# directly (rbfunpack / single-point fits); see the module docstring of RBF below.
#
# Compactly-supported "Gaussian-like" basis phi(u), u = distance / radius.
# ALGLIB's V2 basis is a Gaussian tapered to exactly zero at u = _RBF_CUTOFF.
# The constants below reproduce ALGLIB's basis to < 6e-5 over its whole support.
_RBF_A = 0.93750975
_RBF_D = 2.60513816
_RBF_P = 1.17659512
_RBF_CUTOFF = _RBF_D


def _rbf_phi(u: NDArray[Any]) -> NDArray[Any]:
    """Compact basis function of u = distance / radius (zero for u >= cutoff)."""
    return np.exp(-_RBF_A * u**2) * np.clip(1.0 - (u / _RBF_D) ** 2, 0.0, None) ** _RBF_P


def _rbf_thin(points: NDArray[Any], h: float) -> NDArray[Any]:
    """Greedy poisson-disk decimation: keep a maximal subset of ``points`` whose
    pairwise distance is >= h. Mirrors ALGLIB's per-layer center coarsening,
    where the spacing threshold is h = 0.1 * radius. This is what makes coarse
    (large-radius) layers cheap and the whole scheme ~O(N log N)."""
    if h <= 0 or points.shape[0] == 0:
        return np.arange(points.shape[0])
    tree = cKDTree(points)
    kept_mask = np.zeros(points.shape[0], dtype=bool)
    removed = np.zeros(points.shape[0], dtype=bool)
    kept = []
    for i in range(points.shape[0]):
        if removed[i]:
            continue
        kept.append(i)
        kept_mask[i] = True
        for j in tree.query_ball_point(points[i], h):
            if not kept_mask[j]:
                removed[j] = True
    return np.asarray(kept, dtype=int)


def _rbf_layer_matrix(
    X: NDArray[Any], centers: NDArray[Any], radius: float
) -> sparse.csr_matrix:
    """Sparse (len(X), len(centers)) basis matrix using the compact support of
    phi via a KD-tree, so only the O(1) neighbours within cutoff are evaluated."""
    n_x, n_c = X.shape[0], centers.shape[0]
    if n_c == 0:
        return sparse.csr_matrix((n_x, 0))
    cutoff = _RBF_CUTOFF * radius
    pairs = cKDTree(X).query_ball_tree(cKDTree(centers), r=cutoff)
    rows, cols, vals = [], [], []
    for i, js in enumerate(pairs):
        if not js:
            continue
        js = np.asarray(js)
        d = np.linalg.norm(X[i] - centers[js], axis=1) / radius
        rows.append(np.full(js.shape, i))
        cols.append(js)
        vals.append(_rbf_phi(d))
    if not rows:
        return sparse.csr_matrix((n_x, n_c))
    return sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_x, n_c),
    )


class RBF(Node):
    """
    Radial Basis Function interpolation for 3D LUT generation.

    Pure numpy/scipy reimplementation of ALGLIB's hierarchical RBF
    (``rbfsetalgohierarchical``), so no native ``xalglib`` dependency is needed.

    How it works (reverse-engineered from ALGLIB's V2 model):

    1. A global affine (linear + bias) term is least-squares fitted to the
       source -> target mapping and subtracted, leaving a residual.
    2. A hierarchy of ``layers`` RBF layers fits that residual. Layer ``k`` uses
       radius ``radius * 2**-k`` (the radius halves each layer). The first,
       coarse layers capture broad colour shifts; later, fine layers add detail.
    3. Each layer places centres on a decimated subset of the source points
       (no two centres closer than ``0.1 * radius``), assembles a sparse basis
       matrix from the compactly-supported basis function, and solves for the
       layer weights with a regularised LSQR (``smoothing`` -> Tikhonov damp).
       The layer's contribution is subtracted and the residual passed on.

    The fitted model is baked into a ``size**3`` colour LUT and applied with
    tetrahedral interpolation, exactly as before.

    Based on:
    Lin, Hai Ting, et al. "Nonuniform lattice regression for modeling the
    camera imaging pipeline." ECCV 2012, pp. 556-568.
    """

    def __init__(
        self,
        size: int = 33,
        radius: float = 5.0,
        layers: int = 10,
        smoothing: float = 0.001,
    ):
        """
        Initialize RBF interpolator.

        Args:
            size: Number of nodes per dimension for the output LUT (e.g., 33 = 33³ grid)
            radius: Initial radius for the hierarchical RBF (first/coarsest layer)
            layers: Number of layers in the hierarchical algorithm
            smoothing: Smoothing / regularisation parameter for RBF fitting
        """
        self.size = size
        self.LUT = None
        self.radius = radius
        self.layers = layers
        self.smoothing = smoothing

    def _fit(self, source: NDArray[Any], target: NDArray[Any]) -> None:
        """Fit the hierarchical RBF: affine term + decimated residual layers."""
        X = np.asarray(source, dtype=float)
        Y = np.asarray(target, dtype=float)
        ny = Y.shape[1]

        # Global affine term (matches ALGLIB's default linear polynomial term).
        P = np.hstack([X, np.ones((X.shape[0], 1))])
        coef, *_ = np.linalg.lstsq(P, Y, rcond=None)
        self._coef = coef[:-1].T
        self._bias = coef[-1]
        residual = Y - P @ coef

        # Hierarchical layers over the residual.
        self._source = X
        self._weights, self._radii, self._center_idx = [], [], []
        radius = self.radius
        damp = self.smoothing * np.sqrt(X.shape[0])
        for _ in range(self.layers):
            idx = _rbf_thin(X, 0.1 * radius)
            centers = X[idx]
            A = _rbf_layer_matrix(X, centers, radius)
            w = np.empty((centers.shape[0], ny))
            for c in range(ny):
                w[:, c] = lsqr(A, residual[:, c], damp=damp, atol=1e-10, btol=1e-10)[0]
            residual = residual - A @ w
            self._weights.append(w)
            self._radii.append(radius)
            self._center_idx.append(idx)
            radius *= 0.5

    def _eval(self, RGB: NDArray[Any]) -> NDArray[Any]:
        """Evaluate the fitted hierarchical RBF at arbitrary RGB points."""
        X = np.asarray(RGB, dtype=float)
        out = X @ self._coef.T + self._bias
        for w, r, idx in zip(self._weights, self._radii, self._center_idx):
            out = out + _rbf_layer_matrix(X, self._source[idx], r) @ w
        return out

    def solve(
        self, source: NDArray[Any], target: NDArray[Any]
    ) -> Tuple[NDArray[Any], NDArray[Any]]:
        """
        Fit the RBF model and bake a 3D LUT.

        Args:
            source: Source camera RGB values (N, 3)
            target: Target camera RGB values (N, 3)
        """
        self._fit(source, target)

        # Evaluate the model on the LUT grid. ColourLUT3D indexes the table as
        # [R, G, B], so build the grid in (x, y, z) order directly.
        grid = np.linspace(0, 1, self.size)
        pts = np.stack(
            np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1
        ).reshape(-1, 3)
        LUT_table = self._eval(pts).reshape(self.size, self.size, self.size, 3)

        self.LUT = ColourLUT3D(table=LUT_table)
        return (self(source), target)

    def __call__(self, RGB: NDArray[Any]) -> NDArray[Any]:
        """Apply the RBF-generated LUT to RGB values."""
        if self.LUT is None:
            return RGB

        return self.LUT.apply(RGB, interpolator=table_interpolation_tetrahedral)


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
