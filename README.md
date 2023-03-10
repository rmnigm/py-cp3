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

## How to setup (Unix)
- Clone repository to local machine
- Install Pyenv using [this guide](https://github.com/pyenv/pyenv#installation)
- Install Python, used in project
  ```bash
  $ pyenv install 3.9.16
  ```
  If any problems happen - this [guide](https://github.com/pyenv/pyenv/wiki/Common-build-problems) can help.
- Create virtual environment for Python in repo
  ```bash
  $ cd <path to cloned repo>
  $ ~/.pyenv/versions/3.9.16/bin/python -m venv pycp_env
  ```
- Activate venv (will be active until you clode the terminal session or use `deactivate`)
  ```bash
  $ source pycp_env/bin/activate
  ```  
  In terminal you will now have a prefix:
  ```bash
  (pycp_env)$ ...
  ```

- Check everything is correct and `python` and `pip` lead to `pycp_env`
    ```bash
    (pycp_env)$ which python
    <path to repo>/pycp_env/bin/python
    (pycp_env)$ which pip
    <path to repo>/pycp_env/bin/pip
    ```
- Install dependencies using requirements.txt
  ```bash
  (pycp_env)$ pip install --upgrade -r requirements.txt
  ```
And you are perfect, congratulations!
