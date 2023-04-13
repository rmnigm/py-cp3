import taichi as ti
import taichi_glsl as ts
import time

from utils import draw

ti.init(arch=ti.gpu)

asp = 16 / 9
h = 600
w = int(asp * h)
res = w, h
res_vec = ts.vec2(float(w), float(h))

pixels = ti.Vector.field(3, dtype=ti.f32, shape=res)


@ti.kernel
def run(t: ti.f32):
    bg = ts.vec3(ti.sin(t) * 0.5 + 0.5) * 0.0 + ts.vec3(0.0)
    for frag_coord in ti.grouped(pixels):
        uv = (frag_coord * 2.0 - res_vec) / res_vec[1]
        col = draw(uv, t)
        bg_copy = bg
        pixels[frag_coord] = ts.mix(bg_copy, col.rgb, col.a)


# GUI and main loop

gui = ti.GUI("Taichi simple shader", res=res, fast_gui=True)
start = time.time()

while gui.running:
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == ti.GUI.ESCAPE:
            break
    timer = time.time() - start
    run(timer)
    gui.set_image(pixels)
    gui.show()

gui.close()

