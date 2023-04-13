import taichi as ti
import math

# type shortcuts
vec2 = ti.types.vector(2, ti.f32)
vec3 = ti.types.vector(3, ti.f32)
vec4 = ti.types.vector(4, ti.f32)
mat2 = ti.types.matrix(2, 2, ti.f32)
tmpl = ti.template()

# constants
twopi = 2 * math.pi
pi180 = math.pi / 180.


@ti.func
def clamp(x, low, high):
    return ti.max(ti.min(x, high), low)


@ti.func
def smoothstep(edge0, edge1, x):
    n = (x - edge0) / (edge1 - edge0)
    t = clamp(n, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@ti.func
def plot(r, pct):
    return smoothstep(pct - 0.2, pct, r) - smoothstep(pct, pct + 0.2, r)


@ti.func
def pal(t, a, b, c, d):
    return a + b * ti.cos(twopi * (c * t + d))


@ti.func
def length(p):
    return ti.sqrt(p.dot(p))


@ti.func
def fract(x):
    return x - ti.floor(x)
