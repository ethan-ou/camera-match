# camera_match

`camera_match` is a Python library that provides basic models to match camera colour responses. Using `camera_match`, you can take two cameras with different colour profiles and build a colour pipeline that minimises the difference between them.

Currently, `camera_match` implements the following models:

-   [2D LUT (LUT2D)](https://library.imaging.org/cic/articles/21/1/art00046)
-   [Nonuniform Lattice Regression (LUT3D)](https://link.springer.com/chapter/10.1007/978-3-642-33718-5_40)
-   3x3 Matrix (LinearMatrix)
-   Root Polynomial Matrix (RootPolynomialMatrix)
-   Steve Yedlin's Tetra (TetrahedralMatrix)

## Playground

If you want to use the library without installing anything, I recommend using the Notebook below.

<a href="https://colab.research.google.com/github/ethan-ou/camera-match/blob/main/examples/Camera_Match.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Installation

```bash
pip install camera_match
```

## Camera Matching Methods

### Method 1: 3x3 Matrix + RGB Curves

A 3x3 matrix is a tool that changes the amount of Red, Green, Blue in each channel, as well as how Red interacts with Green and Blue and vice versa. This is a simple and powerful tool that's built into each camera's colour processing.

When digital cameras capture light, they take raw image data, apply a 3x3 matrix then apply a gamma curve (e.g. SLog3, Gamma 2.4). This 3x3 matrix is decided by the camera's engineers and is able to change a number of colour characteristics of the camera. For instance:

-   Red tail lights and skin tones can be bent towards pink or orange
-   Blues can be bent towards navy or cyan
-   Saturation of different channels can be increased by channel

To change this 3x3 matrix, we need to remove our camera's log curve and transform it to linear space. There we can change the matrix that's been given to our liking.

For more on how 3x3 matricies can be used, see Juan Melara's post on [creating a match between Blackmagic cameras and an Arri Alexa](https://juanmelara.com.au/products/bmpcc-6k-to-alexa-powergrade-and-luts).

Benefits of a 3x3 matrix:

-   Linear and exposure invariant
-   Can be applied and adjusted directly in colour grading software (e.g. Resolve's RGB Mixer)
-   Reversible

The flaws of a 3x3 matrix:

-   Can't give control over colour luminance (e.g. make blue darker)
-   Can't create particular hue variations (e.g. can't bend yellow towards orange and blue towards purple)
-   Can't change hue and saturation for many colours e.g. cyan, purple
-   Can't capture non-linear colour responses (e.g. hue bends from green to yellow)
-   Even after correction, can still have major differences in delta E

For more complex situations, a 2D or 3D LUT is necessary to capture colour changes:

### Method 2: 2D LUT + RGB Curves

Paper: [Camera Color Correction Using Two-Dimensional Transforms by Jon S. McElvain, Walter Gish](https://library.imaging.org/cic/articles/21/1/art00046)

A 2D LUT creates a slice of red, green and blue channels and tries to optimise the slice so that colour errors are lower for each channel. Since there's many more degrees of freedom, a 2D LUT is able to capture complex colour shifts and do so across the whole gamut. Overall, a 2D LUT leads to lower colour differences than a 3x3 matrix.

Similar to the 3x3 matrix, a 2D LUT maintains a linear relationship between brightness levels, which means it has fewer drawbacks than a 3D LUT and doesn't clip highlights or shadows. For digital cameras, I'd recommend RGB curves and a 2D LUT for camera matching.

### Method 3: 3D LUT

Paper: [Nonuniform Lattice Regression for Modeling the Camera Imaging Pipeline by Hai Ting Lin, Zheng Lu, Seon Joo Kim & Michael S. Brown](https://link.springer.com/chapter/10.1007/978-3-642-33718-5_40)

LUT3D applies a full 3D colour match between two cameras. This means the colour response is captured across highlight and shadow areas, as well as applying a curve to the image.

This is the most challenging model to gather data for since a lack of data in highlights, shadows and saturated colours can lead to a LUT being clipped early. It's also the model that can break exposure invariance: raising exposure before the LUT can affect the hue, saturation and lightness of colours.

For matching digital to film, this is the model that's able to replicate the most characteristics from the data.

## Gathering Data

For data gathering, I personally use [colour-calibration](https://github.com/ethan-ou/colour-calibration). This is a website which shows thousands of colour patches that I display on an iPad in front of two cameras. I then vary brightness levels on the iPad to gather data across the exposure range.

This creates thousands of colour patches which I then take into Resolve and split into .exr frames. Using a script in Python, I extract the colour from each frame, leading to a .csv file with the colour patches. I can then load this data into camera_match and generate a LUT.

## Examples

### Creating a 3x3 Matrix

A simple matrix that can be used with Resolve's Colour Mixer or any RGB matrix. Can only capture linear changes in colour. We use the Pipeline object and a Colour Space Transform to transform our data into linear.

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
    [CST(source_gamma="S-Log3"), CST(source_gamma="ARRI LogC3")], # Linearises source and target camera data differently.
    LinearMatrix(),
    CST(target_gamma="ARRI LogC3")
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

### Creating a TetrahedralMatrix

Create a match using Steve Yedlin's Tetra.

```python
import numpy as np
from camera_match import TetrahedralMatrix

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

# Create a new TetrahedralMatrix:
matrix = TetrahedralMatrix()

# Find the optimum values to match the two cameras:
matrix.solve(bmpcc_data, film_data)

# Plot the result:
matrix.plot()

# Print the matrix:
print(matrix.matrix)

```

### Creating a LUT using LUT2D or LUT3D

For data-heavy profiling of two cameras.

```python
import numpy as np
from camera_match import LUT2D # For 3D LUT matching, replace this import with LUT3D

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

# Create a new LUT2D or LUT3D node:
lut_2d = LUT2D()

# Find the optimum values to match the two cameras:
lut_2d.solve(bmpcc_data, film_data)

# Plot the result:
lut_2d.plot()

# Export as a LUT:
lut_2d.export_LUT(path="LUT.cube")
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
