import taichi as ti
import taichi_glsl as ts
import time

from utils import plot, length, fract, fluctuate
from utils import fractional_smoothing, periodic_angle_transform, periodic_plane_shift, periodic_plane_scaling

ti.init(arch=ti.gpu)
asp = 16 / 9
h = 800
w = int(asp * h)
res = w, h
resolution_vector = ts.vec2(float(w), float(h))

pixels = ti.Vector.field(3, dtype=ti.f32, shape=res)


# Большая часть изменений внесена в математические функции преобразования координат, основную идею шейдера не трогал.
# Добавил периодические сдвиги центра, другую математическую функцию изменения масштаба,
# поменял обработку угла на другую - добавил логарифм в одну из частей для изменения скорости, но
# оставил модули для наличия прямых отражающих линий в части и оставил временную компоненту для вращения.
# Также поменял функцию перемешивания цветов, заменил её своей после ряда экспериментов
# для получения интересной картинки, сами цвета не стал менять - это просто перебор параметров.


@ti.kernel
def render(t: ti.f32):
    """
    Modified version of the Psychedelic Sakura shader from shadertoy, uses
    periodic functions for plane transformations (e.g. center shifts, scaling,
    angle transformations) and slightly changed color functions and palettes.
    :param t: ti.int32 moment of time for shader rendering
    :return: None
    """
    for frag_coord in ti.grouped(pixels):
        
        # getting the coordinates in center of the picture
        uv = (frag_coord - 0.5 * resolution_vector) / resolution_vector.y
        position = uv
        
        # transforming the plane with several operators
        position = periodic_plane_shift(position, t)
        position = periodic_plane_scaling(position, t)
        
        # getting angle and modified distance to center
        r = length(position) * 3
        a = ts.atan(position.y, position.x)
        
        # periodic angle transformation for spinning and reflecting patterns
        f = periodic_angle_transform(a, t)
        d = f - r
        
        # setting the colors and the patterns of the picture
        color = fractional_smoothing(d, 0.15, 1.15, 0.15)
        color *= fluctuate(f, ts.vec3(0.750, 0.360, 0.352),
                           ts.vec3(0.450, 0.372, 0.271),
                           ts.vec3(0.540, 0.442, 0.264),
                           ts.vec3(0.038, 0.350, 0.107)
                           )
        
        # periodic change for proportion of colors
        pct = plot(r * 0.272, fract(d * (ti.sin(t) * 0.45 + 0.5)))
        
        # second color palette addition into resulting color vector
        color += pct * fluctuate(r, ts.vec3(0.725, 0.475, 0.440),
                                 ts.vec3(0.605, 0.587, 0.007),
                                 ts.vec3(1.0, 1.0, 1.0),
                                 ts.vec3(0.310, 0.410, 0.154)
                                 )
        pixels[frag_coord] = color


# GUI and main loop

gui = ti.GUI("Taichi example shader", res=res, fast_gui=True)
start = time.time()

while gui.running:
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == ti.GUI.ESCAPE:
            break
    timer = time.time() - start
    render(timer)
    gui.set_image(pixels)
    gui.show()

gui.close()
