# Heat Transfer Problem

## Description 
This is the implementation of heat transfer model using **numpy** and **numba**.
Model is based on 2D finite-difference equations of heat transfer. Heat transfer speed depends on diffusion coefficients of objects.
There is also a periodic heat source in one point of the grid and probe points, for which the temperature values are saved as a time series.
Temperature time series in probe points have periodic fluctuations, so trend subtraction and Fast Fourier Transforms is used to define the frequency.

## How to run
* The model can be used interactively in Jupyter Notebook, defined in `heat_transfer_problem_2d.ipynb`.
* If you wish only to use it as a computational model for probe points, you can run it via script with activated **Poetry environment**:
```(bash)
$ poetry run python heat_transfer/heat_transfer.py
```
