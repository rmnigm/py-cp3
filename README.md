# Python for science and engineering
## What is this?
This is a repository containing numerical algorithms in Python for mathematical modelling.
The task author is Stanislav Alekseevich Bober, teacher of [Python course](https://www.hse.ru/edu/courses/646488730) at HSE University.

Tasks are mostly computationally complex problems from physics or mathematical modelling, which require creating a high-performance model and different visualizations. Solutions are annotated in russian language mostly (except docstrings for functions).

**Currently solved:**
- Ferromagnetic energy Izing model
- Lyapunov's fractal interactive map
- Procrustes analysis for sets of 3-dimensional figures
- Heat transfer modelling in 2-dimensional space
- Boids 2D spatial imitational model of agent behavior 

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
