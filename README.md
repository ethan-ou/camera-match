# camera_match

`camera_match` is a Python library that provides basic models to match camera colour responses. Using `camera_match`, you can take two cameras with different colour profiles and build a colour pipeline that minimises the difference between them.

Currently, `camera_match` implements the following models:

-   Linear Colour Correction Matrix
-   Root Polynomial Matrix
-   Steve Yedlin's Tetrahedral Matrix
-   (Experimental) EMoR Response Curves
-   RGB Curve Interpolation
-   Radial Basis Functions

## Installation

(Recommended) Install the full package with the optional RBF library:

```bash
pip install camera_match[RBF]
```

## Examples

### Creating a 3x3 Matrix

```python
import numpy as np
from camera_match import LinearMatrix

# Import samples of a colour chart for your source camera:
bmpcc_data = np.array([
    [0.0460915677249, 0.0414372496307, 0.0392063446343],
    [0.0711114183068, 0.0562727414072, 0.0510282665491],
    [0.0467581525445, 0.0492189191282, 0.0505541190505]
    # ...Additional colour samples
])

# Import corresponding colour patches for your target camera:
film_data = np.array([
    [0.0537128634751, 0.0549002364278, 0.0521950721741],
    [0.0779063776135, 0.0621158666909, 0.0541097335517],
    [0.051306720823, 0.0570512823761, 0.0635398775339]
    # ...Additional colour samples
])

# Create a new LinearMatrix:
matrix = LinearMatrix()

# Find the optimum values to match the two cameras:
matrix.solve(bmpcc_data, film_data)

# Plot the result:
matrix.plot()

# Print the matrix:
print(matrix.matrix)

```

### Creating a LUT using RBF

```python
import numpy as np
from camera_match import RBF

# Import samples of a colour chart for your source camera:
bmpcc_data = np.array([
    [0.0460915677249, 0.0414372496307, 0.0392063446343],
    [0.0711114183068, 0.0562727414072, 0.0510282665491],
    [0.0467581525445, 0.0492189191282, 0.0505541190505]
    # ...Additional colour samples
])

# Import corresponding colour patches for your target camera:
film_data = np.array([
    [0.0537128634751, 0.0549002364278, 0.0521950721741],
    [0.0779063776135, 0.0621158666909, 0.0541097335517],
    [0.051306720823, 0.0570512823761, 0.0635398775339]
    # ...Additional colour samples
])

# Create a new RBF node:
rbf = RBF()

# Find the optimum values to match the two cameras:
rbf.solve(bmpcc_data, film_data)

# Plot the result:
rbf.plot()

# Export as a LUT:
rbf.export_LUT(path="LUT.cube")
```
