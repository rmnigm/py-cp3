import numpy as np
import imageio
from vispy import app, scene  # type: ignore
from vispy.geometry import Rect  # type: ignore
from vispy.scene.visuals import Text  # type: ignore

import config as cfg
from funcs import init_boids, directions, simulation_step


c_names = cfg.coefficients
c = np.array(list(c_names.values()))

boids = np.zeros((cfg.N, 6), dtype=np.float64)
init_boids(boids, cfg.asp, v_range=cfg.v_range)

canvas = scene.SceneCanvas(show=True, size=(cfg.w, cfg.h))
view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(rect=Rect(0, 0, cfg.asp, 1))
arrows = scene.Arrow(arrows=directions(boids, cfg.dt),
                     arrow_color=(1, 1, 1, 1),
                     arrow_size=5,
                     connect='segments',
                     parent=view.scene)
txt = Text(parent=canvas.scene, color='green', face='Consolas')
txt.pos = 15 * canvas.size[0] // 16, canvas.size[1] // 35
txt.font_size = 12
txt_const = Text(parent=canvas.scene, color='green', face='Consolas')
txt_const.pos = canvas.size[0] // 16, canvas.size[1] // 10
txt_const.font_size = 10
general_info = f"boids: {cfg.N}\n"
for key, val in c_names.items():
    general_info += f"{key}: {val}\n"
txt_const.text = general_info


arrows.set_data(arrows=directions(boids, cfg.dt))
writer = imageio.get_writer(f'animation_{cfg.N}.mp4', fps=60)
fr = 0


def make_video(event):
    """Write boids model visualization to video file using VisPy and ffmpeg"""
    global boids, fr, txt
    if fr % 30 == 0:
        txt.text = "fps:" + f"{canvas.fps:0.1f}"
    fr += 1
    simulation_step(boids, cfg.asp, cfg.perception, c, cfg.v_range, cfg.dt)
    arrows.set_data(arrows=directions(boids, cfg.dt))
    if fr <= cfg.frames:
        frame = canvas.render(alpha=False)
        writer.append_data(frame)
    else:
        writer.close()
        app.quit()


def update(event):
    """Render boids model visualization on screen using VisPy"""
    global boids, fr, txt
    if fr % 30 == 0:
        txt.text = "fps:" + f"{canvas.fps:0.1f}"
    fr += 1
    simulation_step(boids, cfg.asp, cfg.perception, c, cfg.v_range, cfg.dt)
    arrows.set_data(arrows=directions(boids, cfg.dt))
    if fr <= cfg.frames:
        canvas.render(alpha=False)
    else:
        app.quit()


if __name__ == '__main__':
    if cfg.video:
        timer = app.Timer(interval=0, start=True, connect=make_video)
    else:
        timer = app.Timer(interval=0, start=True, connect=update)
    canvas.measure_fps()
    app.run()
