# Boids 2D model

## Description
This is the implementation of [boids](https://www.red3d.com/cwr/boids/) model in Python with numpy and numba JIT-compilation. Minor changes were added to formulas of the components - each boid only sees a 180-degree sector in front and some values are medians instead of mean.

Visualization is made with `VisPy` library, `PyQT` video backend and `imageio-ffmpeg` tool.
Example videos are available on [Youtube](https://youtu.be/28eeQrRkj7o).

## How to run
First of all get familiar with `config.py` file - it sets all the useful model parameters.
- Most important are `N` - number of boids - and `coefficients` - values of coefficients for components of acceleration.
- Parameters `video` and `frames` are for visualization - whether to record a video and max frames in video.

After that, you could just run it from root of repository.
```(bash)
(pycp_env)$ python boids/main.py
```

## 