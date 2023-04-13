import taichi as ti
import taichi_glsl as ts
import time

from utils import plot, pal, length, fract, smoothstep

ti.init(arch=ti.gpu)
asp = 16 / 9
h = 600
w = int(asp * h)
res = w, h
resvec = ts.vec2(float(w), float(h))

pixels = ti.Vector.field(3, dtype=ti.f32, shape=res)


@ti.kernel
def render(t: ti.f32):
    for frag_coord in ti.grouped(pixels):
        uv = frag_coord / resvec.xy
        col = ts.vec3(1.0)
        pos = ts.vec2(0.5) - uv
        pos.x *= resvec[0] / resvec[1]
        pos *= ti.cos(t) * 1.0 + 1.5
        
        r = length(pos) * 2.0
        a = ts.atan(pos.y, pos.x)
        
        f = ti.abs(ti.cos(a * 2.5 + t * 0.5)) * ti.sin(t * 2.0) * 0.698 + ti.cos(t) - 4.0
        d = f - r
        
        col = ((ts.vec3(smoothstep(fract(d), fract(d) - 0.200, 0.160))
                - ts.vec3(smoothstep(fract(d), fract(d) - 1.184, 0.160)))
               * pal(f,
                     ts.vec3(0.725, 0.475, 0.440),
                     ts.vec3(0.605, 0.587, 0.007),
                     ts.vec3(1.0, 1.0, 1.0),
                     ts.vec3(0.310, 0.410, 0.154)
                     ))
        
        pct = plot(r * 0.272, fract(d * (ti.sin(t) * 0.45 + 0.5)))
        col += pct * pal(r,
                         ts.vec3(0.750, 0.360, 0.352),
                         ts.vec3(0.450, 0.372, 0.271),
                         ts.vec3(0.540, 0.442, 0.264),
                         ts.vec3(0.038, 0.350, 0.107))
        pixels[frag_coord] = col


# GUI and main loop

gui = ti.GUI("Taichi example shader", res=res, fast_gui=True)
start = time.time()

while gui.running:
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == ti.GUI.ESCAPE:
            break
    t = time.time() - start
    render(t)
    gui.set_image(pixels)
    gui.show()

gui.close()
