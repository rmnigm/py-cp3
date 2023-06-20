# Light Refraction Model

## Description
This is the implementation of light waves refraction model in Python using parallel computations on **Taichi**.
Taichi offers math abstractions such as vectors and different functions and math operators, commonly used in computer graphics.

Light refraction model is based on finite difference equations, which are derived from differential equations of wave refraction proccesses.
Mathematical part of the task is not described here, only resulting equations for a grid of values and open border conditions.

<img src="https://github.com/rmnigm/py-cp3/blob/cf0608507ed546e970d109d7fad81d28542f9336/light_waves/light.gif?raw=true" height="360px" width="640px">

## Structure
* `runner.py` - Taichi kernel, runs the computation on the grid using JIT-compiled code and renders videostream to file or GUI.
* `lens_model.py` - definition of the initial values for model with two symmetrical lenses. Lenses are implemented as intersection of signed distance field masks.
* `utils.py` - implementation of mathemtical operations, used in model initialization and computations.

## How to run
Make sure you have a powerful enough CPU or GPU and set the desired option in `ti.init()` in shader GUI files.
Then just run it from root of repository via **Poetry environment**:
```(bash)
$ poetry run python shaders/runner.py
```
