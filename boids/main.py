import numpy as np
import imageio
from vispy import app, scene  # type: ignore
from vispy.geometry import Rect  # type: ignore
from vispy.scene.visuals import Text  # type: ignore

from funcs import init_boids, directions, simulation_step

w, h = 1920, 1080
N, dt, asp = 100, 0.1, w / h
perception = 1 / 20
v_range = (0, 0.1)
c_names = {"alignment": 0.1, "cohesion": 0.04, "separation": 5, "walls": 0.05}
c = np.array(list(c_names.values()))

boids = np.zeros((N, 6), dtype=np.float64)
init_boids(boids, asp, v_range=v_range)

canvas = scene.SceneCanvas(show=True, size=(w, h))
view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(rect=Rect(0, 0, asp, 1))
arrows = scene.Arrow(arrows=directions(boids, dt),
                     arrow_color=(1, 1, 1, 1),
                     arrow_size=5,
                     connect='segments',
                     parent=view.scene)
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


arrows.set_data(arrows=directions(boids, dt))
writer = imageio.get_writer(f'animation_{N}.mp4', fps=60)
fr = 0


def update(event):
    global boids, fr, txt
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
