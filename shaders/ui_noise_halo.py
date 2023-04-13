import taichi as ti
import taichi_glsl as ts
import time

from utils import light2, light1, smoothstep, extract_alpha, s_noise_3, length, clamp

ti.init(arch=ti.cpu)

asp = 16 / 9
h = 600
w = int(asp * h)
res = w, h
res_vec = ts.vec2(float(w), float(h))

pixels = ti.Vector.field(3, dtype=ti.f32, shape=res)


@ti.func
def draw(vuv, time):
    uv = vuv
    ang = ts.atan(uv.y, uv.x)
    uv_len = length(uv)
    color1 = ts.vec3(0.611765, 0.262745, 0.996078)
    color2 = ts.vec3(0.298039, 0.760784, 0.913725)
    color3 = ts.vec3(0.062745, 0.078431, 0.600000)
    inner_radius = 0.6
    noise_scale = 0.65
    
    n0 = s_noise_3(ts.vec3(uv.x * noise_scale, uv.y * noise_scale, time * 0.5)) * 0.5 + 0.5
    r0 = ts.mix(ts.mix(inner_radius, 1.0, 0.4), ts.mix(inner_radius, 1.0, 0.6), n0)
    d0 = ts.distance(uv, r0 / uv_len * uv)
    v0 = light1(1.0, 10.0, d0)
    v0 *= smoothstep(r0 * 1.05, r0, uv_len)
    cl = ti.cos(ang + time * 2.0) * 0.5 + 0.5
    
    a = time * (-1.0)
    pos = ts.vec2(ti.cos(a), ti.sin(a)) * r0
    d = ts.distance(uv, pos)
    v1 = light2(1.5, 5.0, d)
    v1 *= light1(1.0, 50.0, d0)
    
    v2 = smoothstep(1.0, ts.mix(inner_radius, 1.0, n0 * 0.5), uv_len)
    v3 = smoothstep(inner_radius, ts.mix(inner_radius, 1.0, 0.5), uv_len)
    
    col = ts.mix(color1, color2, cl)
    col = ts.mix(color3, col, v0)
    col = (col + v1) * v2 * v3
    
    col.rgb = clamp(col.rgb, 0.0, 1.)
    return extract_alpha(col)


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

