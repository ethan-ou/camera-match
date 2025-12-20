# Changelog

## 0.0.3

-   Update minimum colour-science version to 0.4.2. In this version the ARRI LogC encoding names have changed.
-   Try fixing issue with Mac importing of xalglib for #1
-   Update docs to show CST and Pipeline nodes.

## 0.1.0

Breaking Changes:

-   Remove RBF and replace with LUT3D and LUT2D. These are based on two papers that use non-linear lattice regression leading to faster and more accurate matches.
-   Fix TetrahedralMatrix to use the same ordering as the DCTL: Red, Green, Blue, Cyan, Magenta, Yellow.

Other Changes:

-   Update Camera Match Jupyter notebook
-   Update to using uv and pyproject.toml for dependencies.
