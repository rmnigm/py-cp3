import numpy as np
import imageio
from vispy import app, scene  # type: ignore
from vispy.geometry import Rect  # type: ignore
from vispy.scene.visuals import Text  # type: ignore

import config as cfg
from funcs import init_boids, directions, simulation_step

if cfg.add_hunters:
    from funcs import init_boids_hunters, simulation_step_hunters


def run_standard_mode():
    """Run the standard boids simulation without hunters."""
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
        nonlocal boids, fr
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
        nonlocal boids, fr
        if fr % 30 == 0:
            txt.text = "fps:" + f"{canvas.fps:0.1f}"
        fr += 1
        simulation_step(boids, cfg.asp, cfg.perception, c, cfg.v_range, cfg.dt)
        arrows.set_data(arrows=directions(boids, cfg.dt))
        if fr <= cfg.frames:
            canvas.render(alpha=False)
        else:
            app.quit()

    if cfg.video:
        timer = app.Timer(interval=0, start=True, connect=make_video)
    else:
        timer = app.Timer(interval=0, start=True, connect=update)
    canvas.measure_fps()
    app.run()


def run_hunters_mode():
    """Run the hunters/prey boids simulation with two agent classes."""
    # Calculate number of prey and predators
    n_predators = max(1, int(cfg.N * cfg.predator_ratio))
    n_prey = cfg.N - n_predators

    # Get coefficients
    prey_c_names = cfg.prey_coefficients
    prey_c = np.array(list(prey_c_names.values()))
    pred_c_names = cfg.predator_coefficients
    pred_c = np.array(list(pred_c_names.values()))

    # Initialize prey and predators
    prey = np.zeros((n_prey, 6), dtype=np.float64)
    predators = np.zeros((n_predators, 6), dtype=np.float64)
    init_boids_hunters(prey, predators, cfg.asp, v_range=cfg.v_range)

    # Create canvas and view
    canvas = scene.SceneCanvas(show=True, size=(cfg.w, cfg.h))
    view = canvas.central_widget.add_view()
    view.camera = scene.PanZoomCamera(rect=Rect(0, 0, cfg.asp, 1))

    # Create arrows for prey (green)
    arrows_prey = scene.Arrow(arrows=directions(prey, cfg.dt),
                              arrow_color=cfg.prey_color,
                              arrow_size=5,
                              connect='segments',
                              parent=view.scene)

    # Create arrows for predators (red)
    arrows_pred = scene.Arrow(arrows=directions(predators, cfg.dt),
                              arrow_color=cfg.predator_color,
                              arrow_size=7,
                              connect='segments',
                              parent=view.scene)

    # FPS text
    txt = Text(parent=canvas.scene, color='green', face='Consolas')
    txt.pos = 15 * canvas.size[0] // 16, canvas.size[1] // 35
    txt.font_size = 12

    # Info text
    txt_const = Text(parent=canvas.scene, color='green', face='Consolas')
    txt_const.pos = canvas.size[0] // 16, canvas.size[1] // 10
    txt_const.font_size = 10

    general_info = f"prey: {n_prey}, predators: {n_predators}\n"
    general_info += f"predator_to_prey: {cfg.predator_to_prey_attraction}\n"
    general_info += f"prey_to_predator: {cfg.prey_to_predator_avoidance}\n"
    general_info += "--- Prey coeffs ---\n"
    for key, val in prey_c_names.items():
        general_info += f"{key}: {val}\n"
    general_info += "--- Predator coeffs ---\n"
    for key, val in pred_c_names.items():
        general_info += f"{key}: {val}\n"
    txt_const.text = general_info

    writer = imageio.get_writer(f'animation_hunters_{cfg.N}.mp4', fps=60)
    fr = 0

    def make_video(event):
        """Write hunters/prey visualization to video file"""
        nonlocal prey, predators, fr
        if fr % 30 == 0:
            txt.text = "fps:" + f"{canvas.fps:0.1f}"
        fr += 1
        simulation_step_hunters(prey, predators, cfg.asp, cfg.perception,
                                prey_c, pred_c,
                                cfg.predator_to_prey_attraction,
                                cfg.prey_to_predator_avoidance,
                                cfg.v_range, cfg.dt)
        arrows_prey.set_data(arrows=directions(prey, cfg.dt))
        arrows_pred.set_data(arrows=directions(predators, cfg.dt))
        if fr <= cfg.frames:
            frame = canvas.render(alpha=False)
            writer.append_data(frame)
        else:
            writer.close()
            app.quit()

    def update(event):
        """Render hunters/prey visualization on screen"""
        nonlocal prey, predators, fr
        if fr % 30 == 0:
            txt.text = "fps:" + f"{canvas.fps:0.1f}"
        fr += 1
        simulation_step_hunters(prey, predators, cfg.asp, cfg.perception,
                                prey_c, pred_c,
                                cfg.predator_to_prey_attraction,
                                cfg.prey_to_predator_avoidance,
                                cfg.v_range, cfg.dt)
        arrows_prey.set_data(arrows=directions(prey, cfg.dt))
        arrows_pred.set_data(arrows=directions(predators, cfg.dt))
        if fr <= cfg.frames:
            canvas.render(alpha=False)
        else:
            app.quit()

    if cfg.video:
        timer = app.Timer(interval=0, start=True, connect=make_video)
    else:
        timer = app.Timer(interval=0, start=True, connect=update)
    canvas.measure_fps()
    app.run()


if __name__ == '__main__':
    if cfg.add_hunters:
        run_hunters_mode()
    else:
        run_standard_mode()
