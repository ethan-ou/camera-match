# camera_match

`camera_match` is a Python library that provides basic models to match camera colour responses. Using `camera_match`, you can take two cameras with different colour profiles and build a colour pipeline that minimises the difference between them.

Currently, `camera_match` implements the following models:

-   Linear Colour Correction Matrix
-   Root Polynomial Matrix
-   Steve Yedlin's Tetrahedral Matrix
-   (Experimental) EMoR Response Curves
-   RGB Curve Interpolation
-   Radial Basis Functions

## Playground

If you want to use the library without installing anything, I recommend using the Notebook below.

<a href="https://colab.research.google.com/github/ethan-ou/camera-match/blob/main/examples/Camera_Match.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Installation

(Recommended) Install the full package with the optional RBF library:

```bash
pip install camera_match[RBF]
```

If you don't need to create LUT's using RBF, you can install the base library:

```bash
pip install camera_match
```

## Examples

### Creating a 3x3 Matrix

A simple matrix that can be used with Resolve's Colour Mixer or any RGB matrix. Can only capture linear changes in colour.

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

Radial Basis Functions (RBF) allows you to create a LUT that smoothly maps your dataset in 3D. This means you can capture complex colour responses that linear matricies can't capture.

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

### Using CST Nodes

Similar to Davinci Resolve, the CST node can be used to transform colour spaces and gammas.

Since this node is just a convenience wrapper around the Colour library, you can use any of the options listed on their docs including [gamma encodings](https://colour.readthedocs.io/en/latest/generated/colour.cctf_decoding.html) and [colour spaces](https://colour.readthedocs.io/en/latest/generated/colour.RGB_COLOURSPACES.html).

```python
# Transform from LogC -> Linear
CST(source_gamma='ARRI LogC3')

# Transform from Linear -> S-Log3
CST(target_gamma="S-Log3")

# Transform from LogC -> SLog3
CST(source_gamma='ARRI LogC3', target_gamma="S-Log3")

# Transform from S-Gamut3.Cine -> Blackmagic Wide Gamut
CST(source_colourspace="S-Gamut3.Cine", target_colourspace="Blackmagic Wide Gamut")

# Combining a gamma and colourspace transform
CST(source_gamma="Blackmagic Film Generation 5", source_colourspace="Blackmagic Wide Gamut", target_gamma='ARRI LogC3', target_colourspace="ARRI Wide Gamut 3")
```

### Building a Pipeline

To create more complex colour pipelines, you can use the Pipeline object to chain multiple nodes together. Here's an example using a LinearMatrix to colour match two digital cameras.

```python
import numpy as np
from camera_match import (
    CST,
    LinearMatrix,
    Pipeline
)

# Import corresponding colour patches for your target camera:
sony_data = np.array([
    [0.0537128634751, 0.0549002364278, 0.0521950721741],
    [0.0779063776135, 0.0621158666909, 0.0541097335517],
    [0.051306720823, 0.0570512823761, 0.0635398775339]
    # ...Additional colour samples
])

# Import samples of a colour chart for your source camera:
alexa_data = np.array([
    [0.0460915677249, 0.0414372496307, 0.0392063446343],
    [0.0711114183068, 0.0562727414072, 0.0510282665491],
    [0.0467581525445, 0.0492189191282, 0.0505541190505]
    # ...Additional colour samples
])

pipeline = Pipeline([
    [CST(source_gamma="S-Log3"), CST(source_gamma='ARRI LogC3')], # Linearises source and target camera data differently.
    LinearMatrix()
])

# Find the optimum values to match the two cameras:
pipeline.solve(sony_data, alexa_data)

# Plot the result:
pipeline.plot()

# Get the matrix:
matrix = pipeline.nodes[1]

# Print the matrix:
print(matrix.matrix)
```
