import numpy as np
import imageio
from vispy import app, scene  # type: ignore
from vispy.geometry import Rect  # type: ignore
from vispy.scene.visuals import Text, Mesh, Ellipse, Markers  # type: ignore

import config as cfg
from funcs import init_boids, directions, simulation_step, angle_to_cos

# Import spatial hashing version if enabled
if cfg.use_spatial_hash:
    from funcs import visibility_spatial as visibility_func
else:
    from funcs import visibility as visibility_func

if cfg.add_hunters:
    from funcs import init_boids_hunters, simulation_step_hunters, visibility_cross

if cfg.add_obstacles:
    from funcs import init_obstacles


def get_max_neighbours_value():
    """Get the max_neighbours value based on config settings."""
    if cfg.use_max_neighbours:
        return cfg.max_neighbours
    return -1  # -1 means no limit


def create_sector_mesh(pos: np.ndarray, speed: np.ndarray, perception: float, 
                       angle_deg: float, n_segments: int = 32):
    """Create vertices and faces for visualizing the visibility sector as a mesh.
    
    Args:
        pos: Position of the agent (x, y)
        speed: Velocity vector of the agent (vx, vy)
        perception: Maximum visibility distance
        angle_deg: Half-angle of the visibility sector in degrees
        n_segments: Number of segments to approximate the arc
        
    Returns:
        Tuple of (vertices, faces) for drawing a filled sector mesh, or (None, None) if speed is zero
    """
    speed_norm = np.sqrt(speed[0]**2 + speed[1]**2)
    
    if speed_norm == 0:
        return None, None
    
    # Direction angle
    dir_angle = np.arctan2(speed[1], speed[0])
    
    # Half-angle in radians
    half_angle = np.radians(angle_deg)
    
    # Generate sector vertices
    angles = np.linspace(dir_angle - half_angle, dir_angle + half_angle, n_segments)
    
    # Vertices: center (index 0) + arc points (indices 1 to n_segments)
    vertices = np.zeros((n_segments + 1, 3))  # 3D for Mesh (z=0)
    vertices[0, :2] = pos  # Center
    
    # Vectorized: compute all arc points at once
    vertices[1:, 0] = pos[0] + perception * np.cos(angles)
    vertices[1:, 1] = pos[1] + perception * np.sin(angles)
    
    # Faces: triangles from center to consecutive arc points
    faces = np.zeros((n_segments - 1, 3), dtype=np.uint32)
    faces[:, 0] = 0  # All triangles start from center
    faces[:, 1] = np.arange(1, n_segments)  # First arc point
    faces[:, 2] = np.arange(2, n_segments + 1)  # Second arc point
    
    return vertices, faces


def create_obstacle_visuals(obstacles: np.ndarray, view) -> list:
    """Create visual representations of obstacles as circles.
    
    Args:
        obstacles: Array of obstacles (n_obstacles x 4) with [x, y, radius, type]
        view: VisPy view to add obstacles to
        
    Returns:
        List of Ellipse visuals for each obstacle
    """
    visuals = []
    for obs in obstacles:
        x, y, radius, obs_type = obs
        # Choose color based on type
        if obs_type == 0:  # Repelling
            color = cfg.obstacle_repel_color
        else:  # Attracting
            color = cfg.obstacle_attract_color
        
        circle = Ellipse(
            center=(x, y),
            radius=radius,
            color=color,
            border_color='white',
            border_width=2,
            parent=view.scene
        )
        visuals.append(circle)
    return visuals


def run_standard_mode():
    """Run the standard boids simulation without hunters."""
    c_names = cfg.coefficients
    c = np.array(list(c_names.values()))
    
    boids = np.zeros((cfg.N, 6), dtype=np.float64)
    init_boids(boids, cfg.asp, v_range=cfg.v_range)

    canvas = scene.SceneCanvas(show=True, size=(cfg.w, cfg.h))
    view = canvas.central_widget.add_view()
    view.camera = scene.PanZoomCamera(rect=Rect(0, 0, cfg.asp, 1))
    
    # Initialize obstacles if enabled
    obstacles = np.zeros((0, 4), dtype=np.float64)
    obstacle_visuals = []
    if cfg.add_obstacles:
        obstacles = init_obstacles(cfg.obstacles, cfg.asp)
        obstacle_visuals = create_obstacle_visuals(obstacles, view)
    
    # Main arrows for all boids
    arrows = scene.Arrow(arrows=directions(boids, cfg.dt),
                         arrow_color=(1, 1, 1, 1),
                         arrow_size=5,
                         connect='segments',
                         parent=view.scene)
    
    # Setup visibility visualization if enabled
    selected_boid_idx = None
    sector_visual = None
    visible_arrows = None
    
    # Setup max_neighbours visualization if enabled
    max_neighbours_markers = None
    visible_not_used_markers = None
    
    if cfg.visualize_angle in ["prey", "predator"]:
        # In standard mode, both options select a random boid
        selected_boid_idx = np.random.randint(0, cfg.N)
        
        # Create sector mesh for visibility area
        sector_visual = Mesh(parent=view.scene, 
                             color=cfg.visibility_area_color)
        
        # Create arrows for visible boids (will be updated each frame)
        visible_arrows = scene.Arrow(arrows=np.zeros((0, 4)),
                                     arrow_color=cfg.visible_boid_color,
                                     arrow_size=6,
                                     connect='segments',
                                     parent=view.scene)
    
    # Setup max_neighbours visualization
    if cfg.visualize_max_neighbours and cfg.use_max_neighbours:
        if selected_boid_idx is None:
            # Select a random boid for visualization
            selected_boid_idx = np.random.randint(0, cfg.N)
        
        # Create markers for max_neighbours (used in calculations) - yellow glow
        max_neighbours_markers = Markers(parent=view.scene)
        max_neighbours_markers.set_data(np.zeros((0, 2)), 
                                        face_color=cfg.max_neighbours_color,
                                        edge_color=None,
                                        size=15,
                                        edge_width=0)
        
        # Create markers for visible but not used neighbours - grey
        visible_not_used_markers = Markers(parent=view.scene)
        visible_not_used_markers.set_data(np.zeros((0, 2)),
                                          face_color=cfg.visible_not_used_color,
                                          edge_color=None,
                                          size=10,
                                          edge_width=0)
    
    # Text displays
    txt = Text(parent=canvas.scene, color='green', face='Consolas')
    txt.pos = 15 * canvas.size[0] // 16, canvas.size[1] // 35
    txt.font_size = 12
    
    txt_const = Text(parent=canvas.scene, color='green', face='Consolas')
    txt_const.pos = canvas.size[0] // 16, canvas.size[1] // 3
    txt_const.font_size = 10
    
    general_info = f"boids: {cfg.N}\n"
    general_info += f"angle: {cfg.angle}°\n"
    for key, val in c_names.items():
        general_info += f"{key}: {val}\n"
    if cfg.use_spatial_hash:
        general_info += f"spatial_hash: enabled\n"
    if cfg.add_obstacles:
        general_info += f"obstacles: {len(obstacles)}\n"
        general_info += f"repel_strength: {cfg.obstacle_repel_strength}\n"
        general_info += f"attract_strength: {cfg.obstacle_attract_strength}\n"
    if cfg.use_max_neighbours:
        general_info += f"max_neighbours: {cfg.max_neighbours}\n"
    if selected_boid_idx is not None:
        general_info += f"visualize: boid #{selected_boid_idx}\n"
    txt_const.text = general_info

    arrows.set_data(arrows=directions(boids, cfg.dt))
    writer = imageio.get_writer(f'animation_{cfg.N}.mp4', fps=60)
    fr = 0

    angle_cos = angle_to_cos(cfg.angle)
    max_neighbours = get_max_neighbours_value()
    
    # Cached visibility masks for visualization (computed once per frame)
    cached_mask_all = None
    cached_mask_limited = None

    def update_visualization():
        """Update all visualizations with cached visibility masks."""
        nonlocal cached_mask_all, cached_mask_limited
        
        # Compute visibility masks once per frame (cached)
        needs_visibility = (selected_boid_idx is not None and sector_visual is not None) or \
                          (max_neighbours_markers is not None and selected_boid_idx is not None)
        
        if needs_visibility:
            cached_mask_all = visibility_func(boids, cfg.perception, angle_cos, -1)
            if max_neighbours > 0:
                cached_mask_limited = visibility_func(boids, cfg.perception, angle_cos, max_neighbours)
            else:
                cached_mask_limited = cached_mask_all
        
        # Update visibility sector visualization
        if selected_boid_idx is not None and sector_visual is not None:
            agent = boids[selected_boid_idx]
            pos = agent[:2]
            speed = agent[2:4]
            
            # Update sector mesh
            vertices, faces = create_sector_mesh(pos, speed, cfg.perception, cfg.angle)
            if vertices is not None and faces is not None:
                sector_visual.set_data(vertices=vertices, faces=faces)
            
            # Use cached mask
            visible_mask = cached_mask_all[selected_boid_idx]
            
            # Update visible arrows
            if np.any(visible_mask):
                visible_directions = directions(boids[visible_mask], cfg.dt)
                visible_arrows.set_data(arrows=visible_directions)
            else:
                visible_arrows.set_data(arrows=np.zeros((0, 4)))
        
        # Update max_neighbours visualization
        if max_neighbours_markers is not None and selected_boid_idx is not None:
            visible_mask = cached_mask_all[selected_boid_idx]
            limited_mask = cached_mask_limited[selected_boid_idx]
            
            # Find neighbours that are visible but not used (beyond max_neighbours)
            visible_but_not_used = visible_mask & ~limited_mask
            
            # Update markers for max_neighbours (used in calculations)
            if np.any(limited_mask):
                positions = boids[limited_mask, :2]
                max_neighbours_markers.set_data(positions,
                                               face_color=cfg.max_neighbours_color,
                                               edge_color=None,
                                               size=15,
                                               edge_width=0)
            else:
                max_neighbours_markers.set_data(np.zeros((0, 2)))
            
            # Update markers for visible but not used
            if np.any(visible_but_not_used):
                positions = boids[visible_but_not_used, :2]
                visible_not_used_markers.set_data(positions,
                                                 face_color=cfg.visible_not_used_color,
                                                 edge_color=None,
                                                 size=10,
                                                 edge_width=0)
            else:
                visible_not_used_markers.set_data(np.zeros((0, 2)))

    def make_video(event):
        """Write boids model visualization to video file using VisPy and ffmpeg"""
        nonlocal boids, fr
        if fr % 30 == 0:
            txt.text = "fps:" + f"{canvas.fps:0.1f}"
        fr += 1
        simulation_step(boids, cfg.asp, cfg.perception, cfg.cohesion_strength, c, cfg.v_range, cfg.dt, 
                       angle_cos, obstacles, 
                       cfg.obstacle_repel_strength, cfg.obstacle_attract_strength,
                       max_neighbours)
        arrows.set_data(arrows=directions(boids, cfg.dt))
        update_visualization()
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
        simulation_step(boids, cfg.asp, cfg.perception, cfg.cohesion_strength, c, cfg.v_range, cfg.dt,
                       angle_cos, obstacles,
                       cfg.obstacle_repel_strength, cfg.obstacle_attract_strength,
                       max_neighbours)
        arrows.set_data(arrows=directions(boids, cfg.dt))
        update_visualization()
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
    
    # Calculate angle cosines for visibility
    angle_cos = angle_to_cos(cfg.angle)
    angle_prey_pred_cos = angle_to_cos(cfg.angle_prey_see_pred)
    angle_pred_prey_cos = angle_to_cos(cfg.angle_pred_see_prey)
    
    # Get max_neighbours value
    max_neighbours = get_max_neighbours_value()

    # Initialize prey and predators
    prey = np.zeros((n_prey, 6), dtype=np.float64)
    predators = np.zeros((n_predators, 6), dtype=np.float64)
    init_boids_hunters(prey, predators, cfg.asp, v_range=cfg.v_range)

    # Create canvas and view
    canvas = scene.SceneCanvas(show=True, size=(cfg.w, cfg.h))
    view = canvas.central_widget.add_view()
    view.camera = scene.PanZoomCamera(rect=Rect(0, 0, cfg.asp, 1))

    # Initialize obstacles if enabled
    obstacles = np.zeros((0, 4), dtype=np.float64)
    obstacle_visuals = []
    if cfg.add_obstacles:
        obstacles = init_obstacles(cfg.obstacles, cfg.asp)
        obstacle_visuals = create_obstacle_visuals(obstacles, view)

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
    
    # Setup visibility visualization if enabled
    selected_prey_idx = None
    selected_pred_idx = None
    sector_visual = None
    visible_prey_arrows = None
    visible_pred_arrows = None
    
    # Setup max_neighbours visualization if enabled
    max_neighbours_markers = None
    visible_not_used_markers = None
    
    if cfg.visualize_angle == "prey":
        selected_prey_idx = np.random.randint(0, n_prey)
        
        # Create sector mesh for visibility area
        sector_visual = Mesh(parent=view.scene,
                             color=cfg.visibility_area_color)
        
        # Create arrows for visible prey (same species)
        visible_prey_arrows = scene.Arrow(arrows=np.zeros((0, 4)),
                                          arrow_color=cfg.visible_prey_color,
                                          arrow_size=6,
                                          connect='segments',
                                          parent=view.scene)
        
        # Create arrows for visible predators (cross-species)
        visible_pred_arrows = scene.Arrow(arrows=np.zeros((0, 4)),
                                          arrow_color=cfg.visible_predator_color,
                                          arrow_size=8,
                                          connect='segments',
                                          parent=view.scene)
        
    elif cfg.visualize_angle == "predator":
        selected_pred_idx = np.random.randint(0, n_predators)
        
        # Create sector mesh for visibility area
        sector_visual = Mesh(parent=view.scene,
                             color=cfg.visibility_area_color)
        
        # Create arrows for visible prey (cross-species)
        visible_prey_arrows = scene.Arrow(arrows=np.zeros((0, 4)),
                                          arrow_color=cfg.visible_prey_color,
                                          arrow_size=6,
                                          connect='segments',
                                          parent=view.scene)
        
        # Create arrows for visible predators (same species)
        visible_pred_arrows = scene.Arrow(arrows=np.zeros((0, 4)),
                                          arrow_color=cfg.visible_predator_color,
                                          arrow_size=8,
                                          connect='segments',
                                          parent=view.scene)
    
    # Setup max_neighbours visualization
    if cfg.visualize_max_neighbours and cfg.use_max_neighbours:
        if selected_prey_idx is None and selected_pred_idx is None:
            # Select a random prey for visualization
            selected_prey_idx = np.random.randint(0, n_prey)
        
        # Create markers for max_neighbours (used in calculations) - yellow glow
        max_neighbours_markers = Markers(parent=view.scene)
        max_neighbours_markers.set_data(np.zeros((0, 2)), 
                                        face_color=cfg.max_neighbours_color,
                                        edge_color=None,
                                        size=15,
                                        edge_width=0)
        
        # Create markers for visible but not used neighbours - grey
        visible_not_used_markers = Markers(parent=view.scene)
        visible_not_used_markers.set_data(np.zeros((0, 2)),
                                          face_color=cfg.visible_not_used_color,
                                          edge_color=None,
                                          size=10,
                                          edge_width=0)

    # FPS text
    txt = Text(parent=canvas.scene, color='green', face='Consolas')
    txt.pos = 15 * canvas.size[0] // 16, canvas.size[1] // 35
    txt.font_size = 12

    # Info text
    txt_const = Text(parent=canvas.scene, color='green', face='Consolas')
    txt_const.pos = canvas.size[0] // 16, canvas.size[1] // 3
    txt_const.font_size = 10

    general_info = f"prey: {n_prey}, predators: {n_predators}\n"
    general_info += f"angle: {cfg.angle}°\n"
    general_info += f"angle_prey_see_pred: {cfg.angle_prey_see_pred}°\n"
    general_info += f"angle_pred_see_prey: {cfg.angle_pred_see_prey}°\n"
    general_info += f"predator_to_prey: {cfg.predator_to_prey_attraction}\n"
    general_info += f"prey_to_predator: {cfg.prey_to_predator_avoidance}\n"
    if cfg.use_spatial_hash:
        general_info += f"spatial_hash: enabled\n"
    if cfg.add_obstacles:
        general_info += f"obstacles: {len(obstacles)}\n"
        general_info += f"repel_strength: {cfg.obstacle_repel_strength}\n"
        general_info += f"attract_strength: {cfg.obstacle_attract_strength}\n"
    if cfg.use_max_neighbours:
        general_info += f"max_neighbours: {cfg.max_neighbours}\n"
    general_info += "--- Prey coeffs ---\n"
    for key, val in prey_c_names.items():
        general_info += f"{key}: {val}\n"
    general_info += "--- Predator coeffs ---\n"
    for key, val in pred_c_names.items():
        general_info += f"{key}: {val}\n"
    if selected_prey_idx is not None:
        general_info += f"visualize: prey #{selected_prey_idx}\n"
    elif selected_pred_idx is not None:
        general_info += f"visualize: predator #{selected_pred_idx}\n"
    txt_const.text = general_info

    writer = imageio.get_writer(f'animation_hunters_{cfg.N}.mp4', fps=60)
    fr = 0

    def update_visualization():
        """Update all visualizations for hunters mode with optimized visibility calls."""
        # Determine what visualizations are needed
        needs_sector = sector_visual is not None
        needs_max_neighbours = max_neighbours_markers is not None
        
        if selected_prey_idx is not None:
            # Visualizing a prey agent
            agent = prey[selected_prey_idx]
            pos = agent[:2]
            speed = agent[2:4]
            
            # Update sector mesh if needed
            if needs_sector:
                vertices, faces = create_sector_mesh(pos, speed, cfg.perception, cfg.angle)
                if vertices is not None and faces is not None:
                    sector_visual.set_data(vertices=vertices, faces=faces)
            
            # Compute visibility masks once (cached for this frame)
            mask_prey_all = visibility_func(prey, cfg.perception, angle_cos, -1)
            visible_prey_mask = mask_prey_all[selected_prey_idx]
            
            if max_neighbours > 0:
                mask_prey_limited = visibility_func(prey, cfg.perception, angle_cos, max_neighbours)
                limited_prey_mask = mask_prey_limited[selected_prey_idx]
            else:
                limited_prey_mask = visible_prey_mask
            
            # Cross-species visibility
            mask_pred_all = visibility_cross(prey[selected_prey_idx:selected_prey_idx+1], 
                                            predators, cfg.perception, angle_prey_pred_cos, -1)
            visible_pred_mask = mask_pred_all[0]
            
            if max_neighbours > 0:
                mask_pred_limited = visibility_cross(prey[selected_prey_idx:selected_prey_idx+1],
                                                    predators, cfg.perception, angle_prey_pred_cos, max_neighbours)
                limited_pred_mask = mask_pred_limited[0]
            else:
                limited_pred_mask = visible_pred_mask
            
            # Update visible arrows
            if needs_sector:
                if np.any(visible_prey_mask):
                    visible_directions = directions(prey[visible_prey_mask], cfg.dt)
                    visible_prey_arrows.set_data(arrows=visible_directions)
                else:
                    visible_prey_arrows.set_data(arrows=np.zeros((0, 4)))
                
                if np.any(visible_pred_mask):
                    visible_directions = directions(predators[visible_pred_mask], cfg.dt)
                    visible_pred_arrows.set_data(arrows=visible_directions)
                else:
                    visible_pred_arrows.set_data(arrows=np.zeros((0, 4)))
            
            # Update max_neighbours markers
            if needs_max_neighbours:
                visible_but_not_used_prey = visible_prey_mask & ~limited_prey_mask
                visible_but_not_used_pred = visible_pred_mask & ~limited_pred_mask
                
                positions_list = []
                if np.any(limited_prey_mask):
                    positions_list.append(prey[limited_prey_mask, :2])
                if np.any(limited_pred_mask):
                    positions_list.append(predators[limited_pred_mask, :2])
                
                if positions_list:
                    positions = np.vstack(positions_list)
                    max_neighbours_markers.set_data(positions,
                                                   face_color=cfg.max_neighbours_color,
                                                   edge_color=None,
                                                   size=15,
                                                   edge_width=0)
                else:
                    max_neighbours_markers.set_data(np.zeros((0, 2)))
                
                positions_list = []
                if np.any(visible_but_not_used_prey):
                    positions_list.append(prey[visible_but_not_used_prey, :2])
                if np.any(visible_but_not_used_pred):
                    positions_list.append(predators[visible_but_not_used_pred, :2])
                
                if positions_list:
                    positions = np.vstack(positions_list)
                    visible_not_used_markers.set_data(positions,
                                                     face_color=cfg.visible_not_used_color,
                                                     edge_color=None,
                                                     size=10,
                                                     edge_width=0)
                else:
                    visible_not_used_markers.set_data(np.zeros((0, 2)))
                
        elif selected_pred_idx is not None:
            # Visualizing a predator agent
            agent = predators[selected_pred_idx]
            pos = agent[:2]
            speed = agent[2:4]
            
            # Update sector mesh if needed
            if needs_sector:
                vertices, faces = create_sector_mesh(pos, speed, cfg.perception, cfg.angle)
                if vertices is not None and faces is not None:
                    sector_visual.set_data(vertices=vertices, faces=faces)
            
            # Compute visibility masks once (cached for this frame)
            mask_pred_all = visibility_func(predators, cfg.perception, angle_cos, -1)
            visible_pred_mask = mask_pred_all[selected_pred_idx]
            
            if max_neighbours > 0:
                mask_pred_limited = visibility_func(predators, cfg.perception, angle_cos, max_neighbours)
                limited_pred_mask = mask_pred_limited[selected_pred_idx]
            else:
                limited_pred_mask = visible_pred_mask
            
            # Cross-species visibility
            mask_prey_all = visibility_cross(predators[selected_pred_idx:selected_pred_idx+1],
                                            prey, cfg.perception, angle_pred_prey_cos, -1)
            visible_prey_mask = mask_prey_all[0]
            
            if max_neighbours > 0:
                mask_prey_limited = visibility_cross(predators[selected_pred_idx:selected_pred_idx+1],
                                                    prey, cfg.perception, angle_pred_prey_cos, max_neighbours)
                limited_prey_mask = mask_prey_limited[0]
            else:
                limited_prey_mask = visible_prey_mask
            
            # Update visible arrows
            if needs_sector:
                if np.any(visible_prey_mask):
                    visible_directions = directions(prey[visible_prey_mask], cfg.dt)
                    visible_prey_arrows.set_data(arrows=visible_directions)
                else:
                    visible_prey_arrows.set_data(arrows=np.zeros((0, 4)))
                
                if np.any(visible_pred_mask):
                    visible_directions = directions(predators[visible_pred_mask], cfg.dt)
                    visible_pred_arrows.set_data(arrows=visible_directions)
                else:
                    visible_pred_arrows.set_data(arrows=np.zeros((0, 4)))
            
            # Update max_neighbours markers
            if needs_max_neighbours:
                visible_but_not_used_prey = visible_prey_mask & ~limited_prey_mask
                visible_but_not_used_pred = visible_pred_mask & ~limited_pred_mask
                
                positions_list = []
                if np.any(limited_prey_mask):
                    positions_list.append(prey[limited_prey_mask, :2])
                if np.any(limited_pred_mask):
                    positions_list.append(predators[limited_pred_mask, :2])
                
                if positions_list:
                    positions = np.vstack(positions_list)
                    max_neighbours_markers.set_data(positions,
                                                   face_color=cfg.max_neighbours_color,
                                                   edge_color=None,
                                                   size=15,
                                                   edge_width=0)
                else:
                    max_neighbours_markers.set_data(np.zeros((0, 2)))
                
                positions_list = []
                if np.any(visible_but_not_used_prey):
                    positions_list.append(prey[visible_but_not_used_prey, :2])
                if np.any(visible_but_not_used_pred):
                    positions_list.append(predators[visible_but_not_used_pred, :2])
                
                if positions_list:
                    positions = np.vstack(positions_list)
                    visible_not_used_markers.set_data(positions,
                                                     face_color=cfg.visible_not_used_color,
                                                     edge_color=None,
                                                     size=10,
                                                     edge_width=0)
                else:
                    visible_not_used_markers.set_data(np.zeros((0, 2)))

    def make_video(event):
        """Write hunters/prey visualization to video file"""
        nonlocal prey, predators, fr
        if fr % 30 == 0:
            txt.text = "fps:" + f"{canvas.fps:0.1f}"
        fr += 1
        simulation_step_hunters(prey, predators, cfg.asp, cfg.perception, cfg.cohesion_strength,
                                prey_c, pred_c,
                                cfg.predator_to_prey_attraction,
                                cfg.prey_to_predator_avoidance,
                                cfg.v_range, cfg.dt,
                                angle_cos, angle_prey_pred_cos, angle_pred_prey_cos,
                                obstacles, cfg.obstacle_repel_strength, cfg.obstacle_attract_strength,
                                max_neighbours)
        arrows_prey.set_data(arrows=directions(prey, cfg.dt))
        arrows_pred.set_data(arrows=directions(predators, cfg.dt))
        update_visualization()
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
        simulation_step_hunters(prey, predators, cfg.asp, cfg.perception, cfg.cohesion_strength,
                                prey_c, pred_c,
                                cfg.predator_to_prey_attraction,
                                cfg.prey_to_predator_avoidance,
                                cfg.v_range, cfg.dt,
                                angle_cos, angle_prey_pred_cos, angle_pred_prey_cos,
                                obstacles, cfg.obstacle_repel_strength, cfg.obstacle_attract_strength,
                                max_neighbours)
        arrows_prey.set_data(arrows=directions(prey, cfg.dt))
        arrows_pred.set_data(arrows=directions(predators, cfg.dt))
        update_visualization()
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
