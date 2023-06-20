# Interactive map for Lyapunov's fractal

## Description 
* Fractals are geometric shapes containing detailed structure at arbitrarily small scales. Some of them are defined as an extension of different maps.
* In this notebook, interactive map is used to explore the [Lyapunov Fractal](https://en.wikipedia.org/wiki/Lyapunov_fractal).
* The fractal image is calculated with set precision each time the map is moved, so you can explore the fractal without needing it be ready ahead of time.

<img src="https://github.com/rmnigm/py-cp3/blob/a127e9d632da4670ef6a536b1521fbae8e8f6880/fractals/fractal.png" width=700>

## Hot to run
* Just run the notebook with **Poetry Python kernel**: 
```(bash)
- poetry run jupyter notebook
```
* If errors occur, the best idea is to clear the viewer cache for JS-based **holoviews** and update the page.
