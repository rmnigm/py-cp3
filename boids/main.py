import numpy as np
import imageio
from datetime import datetime
from vispy import app, scene
from vispy.geometry import Rect
from vispy.scene.visuals import Text

from funcs import init_boids, directions, simulation_step

w, h = 1920, 1080
N = 100
dt = 0.1
asp = w / h
perception = 1 / 20
v_range = (0, 0.1)
c_names = {"alignment": 0.1, "cohesion": 0.04, "separation": 5, "walls": 0.05}
c = np.array(list(c_names.values()))

boids = np.zeros((N, 6), dtype=np.float64)
init_boids(boids, asp, v_range=v_range)
fr = 0

canvas = scene.SceneCanvas(show=True, size=(w, h))
view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(rect=Rect(0, 0, asp, 1))
arrows = scene.Arrow(arrows=directions(boids, dt),
                     arrow_color=(1, 1, 1, 1),
                     arrow_size=5,
                     connect='segments',
                     parent=view.scene)
arrows.set_data(arrows=directions(boids, dt))

txt = Text(parent=canvas.scene, color='green', face='Consolas')
txt.pos = canvas.size[0] // 16, canvas.size[1] // 35
txt.font_size = 12
txt_const = Text(parent=canvas.scene, color='green', face='Consolas')
txt_const.pos = canvas.size[0] // 16, canvas.size[1] // 10
txt_const.font_size = 10
general_info = f"boids: {N}\n"
for key, val in c_names.items():
    general_info += f"{key}: {val}\n"
txt_const.text = general_info


fname = f"boids_{datetime.now().strftime('%Y-%m-%dT%H-%M')}.mp4"
print(fname)

# process = ffmpeg\
#     .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f"{w}x{h}", r=60)\
#     .output(fname, pix_fmt='yuv420p', preset='slower', r=60)\
#     .overwrite_output()\
#     .run_async(pipe_stdin=True)
writer = imageio.get_writer(f'animation_{N}.mp4', fps=60)


def update(event):
    global boids, fr, txt
    # global process
    if fr % 30 == 0:
        txt.text = "fps:" + f"{canvas.fps:0.1f}"
    fr += 1
    simulation_step(boids, asp, perception, c, v_range, dt)
    arrows.set_data(arrows=directions(boids, dt))
    if fr <= 1800:
        frame = canvas.render(alpha=False)
        writer.append_data(frame)
    else:
        writer.close()
        app.quit()


if __name__ == '__main__':
    timer = app.Timer(interval=0, start=True, connect=update)
    canvas.measure_fps()
    app.run()
