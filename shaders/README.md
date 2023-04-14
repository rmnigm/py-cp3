# Shaders

## Description
This is the implementation of two different shaders from [Shadertoy](https://www.shadertoy.com) in Python using **Taichi**.
**Taichi** an imperative, parallel programming language for high-performance numerical computation, which uses JIT-compilation.

Taichi offers math abstractions such as vectors and different functions and math operators, commonly used in computer graphics.
Some are reimplemented in this small project and defined in `utils.py`.

Shaders themselves are just certain mathematical transformations to 2D plane and objects with colors, changing periodically in time.
* `sakura` implements periodic plane shifts, scaling, spins and reflections.
* `ui_noise_halo` implements a normal circle, which can be changed to circle of any metric, and random noises for fluctuations around that circle.

## UI Noise Halo shader
<img src="https://github.com/rmnigm/py-cp3/blob/84d14af3215326c199e3de8dd29fa45a3a3361f1/shaders/noisy_halo.gif?raw=true" height="360px" width="640px">

## How to run
Make sure you have a powerful enough CPU or GPU and set the desired option in `ti.init()` in shader GUI files.
Then just run it from root of repository via **Poetry environment**:
```(bash)
$ poetry run python shaders/ui_noise_halo.py
```

##
