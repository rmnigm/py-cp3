import taichi as ti
import taichi_glsl as ts
import numpy as np

from triangular_lens import setup


ti.init(arch=ti.cpu)
nx, ny = 800, 450
res = nx, ny
res_vec = ts.vec2(float(nx), float(ny))

h = 1.0             # пространственный шаг решетки
c = 1.0             # скорость распространения волн
dt = h / (c * 1.5)  # временной шаг
acc = 0.1           # вес кадра для аккумулятора
n = np.array([      # коэффициент преломления
    1.30,           # R
    1.35,           # G
    1.40            # B
])


past = ti.Vector.field(3, dtype=ti.f32, shape=res)
present = ti.Vector.field(3, dtype=ti.f32, shape=res)
future = ti.Vector.field(3, dtype=ti.f32, shape=res)
ahead = ti.Vector.field(3, dtype=ti.f32, shape=res)
kappa = ti.Vector.field(3, dtype=ti.f32, shape=res)


pixels = ti.Vector.field(3, dtype=ti.f32, shape=res)

model = setup(nx, ny)
kappa.from_numpy(model['kappa'])
past.from_numpy(model['field'])
present.from_numpy(model['field'])
future.from_numpy(model['field'])
ahead.from_numpy(model['field'])


@ti.func
def propagate():
    """
    Один шаг интегрирования уравнений распространения волны по Эйлеру
    """
    for x, y in ahead:
        if x != 0 and y != 0 and x != nx and y != ny:
            ahead[x, y] = kappa[x, y] ** 2 * (
                future[x - 1, y] +
                future[x + 1, y] +
                future[x, y - 1] +
                future[x, y + 1] -
                4 * future[x, y]
            ) + 2 * future[x, y] - present[x, y]


@ti.func
def time_shift():
    for x, y in future:
        if x != 0 and y != 0 and x != nx and y != ny:
            past[x, y] = present[x, y]
            present[x, y] = future[x, y]
            future[x, y] = ahead[x, y]


@ti.func
def open_boundary():
    """
    Граничные условия открытой границы
    """
    for i, j in future:
        if i == 0:
            future[i, j] = (present[i + 1, j]
                            + (kappa[i, j] - 1) / (kappa[i, j] + 1)
                            * (future[i + 1, j] - present[i, j])
                            )
        elif i == nx - 1:
            future[i, j] = (present[i - 1, j]
                            + (kappa[i, j] - 1) / (kappa[i, j] + 1)
                            * (future[i - 1, j] - present[i, j])
                            )
        if j == 0:
            future[i, j] = (present[i, j + 1]
                            + (kappa[i, j] - 1) / (kappa[i, j] + 1)
                            * (future[i, j + 1] - present[i, j])
                            )
        elif j == ny - 1:
            future[i, j] = (present[i, j - 1]
                            + (kappa[i, j] - 1) / (kappa[i, j] + 1)
                            * (future[i, j - 1] - present[i, j])
                            )


@ti.func
def accumulate():
    for i, j in pixels:
        if 0 < i < nx - 1 and 0 < j < ny - 1:
            pixels[i, j] += acc * ti.abs(ahead[i, j]) * kappa[i, j] / (c * dt / h)
            # pixels[i, j] = kappa[i, j]


@ti.kernel
def render():
    open_boundary()
    propagate()
    time_shift()
    accumulate()


gui = ti.GUI("Light Refraction", res=res, fast_gui=True)

while gui.running:
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == ti.GUI.ESCAPE:
            break
    render()
    gui.set_image(pixels)
    gui.show()

gui.close()
