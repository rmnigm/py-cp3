# Python for science and engineering
## What is this?
This is a repository containing numerical algorithms in Python for mathematical modelling.
The task author is Stanislav Alekseevich Bober, teacher of [Python course](https://www.hse.ru/edu/courses/646488730) at HSE University.

Tasks are mostly computationally complex problems from physics or mathematical modelling, which require creating a high-performance model and different visualizations. And they are really pretty!

<img src="https://github.com/rmnigm/py-cp3/blob/84d14af3215326c199e3de8dd29fa45a3a3361f1/shaders/noisy_halo.gif?raw=true" height="360px" width="640px">

<img src="https://github.com/rmnigm/py-cp3/blob/cf0608507ed546e970d109d7fad81d28542f9336/light_waves/light.gif?raw=true" height="360px" width="640px">

## Models and implementations
- Light waves refraction with **Taichi**
- Shaders and computer graphics with **Taichi**
- Boids 2D spatial imitational model of agent behavior
- Heat transfer modelling in 2-dimensional space
- Ferromagnetic energy Izing model
- Lyapunov's fractal interactive map
- Procrustes analysis for sets of 3-dimensional figures

## How to setup (Unix)
- Clone repository to local machine
- Install Pyenv using [this guide](https://github.com/pyenv/pyenv#installation)
- Install [Poetry](https://python-poetry.org)
- Install Python, used in project:
  ```bash
  $ pyenv install 3.9.16
  ```
  If any problems happen - this [guide](https://github.com/pyenv/pyenv/wiki/Common-build-problems) can help.

- Install everything via `poetry install`:
  ```bash
  $ poetry install
  ```
- Use Poetry tools to run scripts and configure dependencies:
  ```bash
  $ poetry run jupyter lab
  ```

And you are perfect, congratulations!
