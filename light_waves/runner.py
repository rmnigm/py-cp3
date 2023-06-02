import taichi as ti
import taichi_glsl as ts
import numpy as np
from tqdm import tqdm

from lens_model import setup


ti.init(arch=ti.cpu)
nx, ny = 1600, 901
res = nx, ny
res_vec = ts.vec2(float(nx), float(ny))

h = 1.0             # пространственный шаг решетки
c = 1.0             # скорость распространения волн
dt = h / (c * 1.5)  # временной шаг
acc = 0.1           # вес кадра для аккумулятора
n = np.array([      # коэффициент преломления
    1.20,           # R
    1.25,           # G
    1.30            # B
])


light = ti.Struct.field({
    "past": ti.math.vec3,
    "present": ti.math.vec3,
    "future": ti.math.vec3,
    "ahead": ti.math.vec3,
    }, shape=res)

kappa = ti.Vector.field(3, dtype=ti.f32, shape=res)
pixels = ti.Vector.field(3, dtype=ti.f32, shape=res)

model = setup(nx, ny)
filler = {
    'past': model['field'],
    'present': model['field'],
    'future': model['field'],
    'ahead': model['field'],
}
kappa.from_numpy(model['kappa'])
light.from_numpy(filler)


@ti.func
def propagate():
    """One step of integrating the Euler equations for numerical refraction modelling"""
    for x, y in light:
        light[x, y].ahead = kappa[x, y] ** 2 * (
                light[x - 1, y].future +
                light[x + 1, y].future +
                light[x, y - 1].future +
                light[x, y + 1].future -
                4 * light[x, y].future
        ) + 2 * light[x, y].future - light[x, y].present


@ti.func
def time_shift():
    """Shift arrays of points for one step forward in time"""
    for x, y in light:
        light[x, y].past = light[x, y].present
        light[x, y].present = light[x, y].future
        light[x, y].future = light[x, y].ahead


@ti.func
def open_boundary():
    """Boundary conditions of open boundary for model of wave refraction"""
    for i, j in light:
        if i == 0:
            light[i, j].future = (light[i + 1, j].present
                                  + (kappa[i, j] - 1) / (kappa[i, j] + 1)
                                  * (light[i + 1, j].present - light[i, j].present)
                                  )
        elif i == nx - 1:
            light[i, j].future = (light[i - 1, j].present
                                  + (kappa[i, j] - 1) / (kappa[i, j] + 1)
                                  * (light[i - 1, j].future - light[i, j].present)
                                  )
        if j == 0:
            light[i, j].future = (light[i, j + 1].present
                                  + (kappa[i, j] - 1) / (kappa[i, j] + 1)
                                  * (light[i, j + 1].future - light[i, j].present)
                                  )
        elif j == ny - 1:
            light[i, j].future = (light[i, j - 1].present
                                  + (kappa[i, j] - 1) / (kappa[i, j] + 1)
                                  * (light[i, j - 1].future - light[i, j].present)
                                  )


@ti.func
def accumulate():
    """Accumulate the light movement in time and visualize it via Taichi GUI"""
    for i, j in pixels:
        if 0 < i < nx - 1 and 0 < j < ny - 1:
            pixels[i, j] += acc * ti.abs(light[i, j].future) * kappa[i, j] / (c * dt / h)
            # pixels[i, j] = kappa[i, j]


@ti.kernel
def render():
    """Kernel for refraction model with Euler equations, implemented in Taichi"""
    open_boundary()
    propagate()
    time_shift()
    accumulate()


# video_manager = ti.tools.VideoManager(output_dir="./angled_by_axis", framerate=60, automatic_build=False)
#
# for i in tqdm(range(2700)):
#     render()
#     img = pixels.to_numpy()[:, :-1, :]
#     video_manager.write_frame(img)
#
# video_manager.make_video(gif=False, mp4=True)

gui = ti.GUI("Light Refraction", res=res, fast_gui=True)

while gui.running:
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == ti.GUI.ESCAPE:
            break
    render()
    gui.set_image(pixels)
    gui.show()
gui.close()
