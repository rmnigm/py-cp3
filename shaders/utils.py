import taichi as ti
import taichi_glsl as ts
import math

# type shortcuts
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


@ti.func
def hash33(p):
    p = ts.fract(p * ts.vec3(0.1031, 0.11369, 0.13787))
    p += p.dot(p.yzx + 19.19)
    return -1.0 + 2.0 * ts.fract((p.xxy + p.yyz) * p.zyx)


@ti.func
def s_noise_3(p):
    k1 = 0.333333333
    k2 = 0.166666667
    
    i = ti.floor(p + (p.x + p.y + p.z) * k1)
    d0 = p - (i - (i.x + i.y + i.z) * k2)
    
    e = ts.step(ts.vec3(0.0), d0 - ts.vec3(d0.y, d0.z, d0.x))
    i1 = e * (1.0 - e.zxy)
    i2 = 1.0 - e.zxy * (1.0 - e)
    
    d1 = d0 - (i1 - k2)
    d2 = d0 - (i2 - k1)
    d3 = d0 - 0.5
    
    h = ti.max(0.6 - ts.vec4(d0.dot(d0), d1.dot(d1), d2.dot(d2), d3.dot(d3)), 0.0)
    n = h * h * h * h * ts.vec4(d0.dot(hash33(i)),
                                d1.dot(hash33(i + i1)),
                                d2.dot(hash33(i + i2)),
                                d3.dot(hash33(i + 1.0))
                                )
    return ts.vec4(31.316) @ n


@ti.func
def extract_alpha(color_in):
    color_out = ts.vec4(0.0)
    max_value = ti.min(ti.max(ti.max(color_in.r, color_in.g), color_in.b), 1.0)
    if max_value > 1e-5:
        color_out.rgb = color_in.rgb * (1.0 / max_value)
        color_out.a = max_value
    else:
        color_out = ts.vec4(0.0)
    return color_out


@ti.func
def light1(intensity, attenuation, dist):
    return intensity / (1.0 + dist * attenuation)


@ti.func
def light2(intensity, attenuation, dist):
    return intensity / (1.0 + dist * dist * attenuation)


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
