w = 1920
h = 1080
N = 100
dt = 0.1
asp = w / h
perception = 1 / 20 # Perception radius (in normalized units, relative to window size)
cohesion_strength = 0.5

v_range = (0, 0.1)
coefficients = {"alignment": 0.3,
                "cohesion": 0.5,
                "separation": 15,
                "walls": 0.2,
                "noise": 0.04}

video = False
frames = 3600

# Variant 1: Hunters (predators) and Prey
add_hunters = False
predator_ratio = 1 / 10  # Predators are 10x fewer than prey

# Prey coefficients (all interactions active)
prey_coefficients = {"alignment": 0.3,
                     "cohesion": 0.5,
                     "separation": 15,
                     "walls": 0.2,
                     "noise": 0.04}

# Predator coefficients (reduced alignment/cohesion, reduced separation to chase prey)
predator_coefficients = {"alignment": 0.05,   # Strongly reduced - independent
                         "cohesion": 0.05,    # Strongly reduced - independent
                         "separation": 15,
                         "walls": 0.2,
                         "noise": 0.04}

# Cross-species interaction coefficients
predator_to_prey_attraction = 2  # Predators attracted to prey
prey_to_predator_avoidance = 35   # Prey avoid predators

# Colors for visualization (RGBA tuples, 0-1 range)
prey_color = (0, 1, 0, 1)       # Green for prey
predator_color = (1, 0, 0, 1)   # Red for predators

# Variant 3: Sector-based visibility angles (in degrees, 90-270)
# The angle defines the half-angle of the visibility sector (total sector = 2 * angle)
angle = 90                     # General visibility angle for same-species interactions
angle_prey_see_pred = 90       # Angle for prey seeing predators
angle_pred_see_prey = 90       # Angle for predators seeing prey

# Visualization mode for visibility area
# Options: "off" - disabled, "prey" - show for random prey, "predator" - show for random predator
# Note: In standard mode (without hunters), "prey" or "predator" will show for a random boid
visualize_angle = "off"

# Colors for visibility visualization (RGBA tuples, 0-1 range)
visibility_area_color = (0.5, 0.5, 1.0, 0.4)      # Light blue, semi-transparent
visible_prey_color = (0.8, 0.6, 1.0, 1.0)          # Light purple for visible prey
visible_predator_color = (1.0, 0.6, 0.8, 1.0)      # Pink for visible predators
visible_boid_color = (0.8, 0.6, 1.0, 1.0)          # Light purple for visible boids (standard mode)

# Variant 2: Obstacles
add_obstacles = True  # Enable/disable obstacles feature

# Obstacles defined as list of (radius, type) pairs
# type 0 = repelling (pushes agents away), 1 = attracting (pulls agents in)
# Positions are generated randomly at initialization
obstacles = [
    (0.05, 0),   # Small repelling obstacle
    (0.08, 1),   # Larger attracting obstacle
    (0.04, 0),   # Another small repelling obstacle
    (0.06, 1),   # Medium attracting obstacle
]

# Obstacle interaction strength
obstacle_repel_strength = 0.2    # Strength of repelling force
obstacle_attract_strength = 0.01   # Strength of attracting force

# Colors for obstacle visualization (RGBA tuples, 0-1 range)
obstacle_repel_color = (1.0, 0.3, 0.3, 0.5)    # Red for repelling obstacles
obstacle_attract_color = (0.3, 0.7, 1.0, 0.5)  # Blue for attracting obstacles
