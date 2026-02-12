w = 1920
h = 1080
N = 100
dt = 0.1
asp = w / h
perception = 1 / 20
v_range = (0, 0.1)
coefficients = {"alignment": 0.3,
                "cohesion": 0.5,
                "separation": 15,
                "walls": 0.2,
                "noise": 0.04}

video = False
frames = 7200

# Variant 1: Hunters (predators) and Prey
add_hunters = True
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
predator_to_prey_attraction = 5  # Predators attracted to prey
prey_to_predator_avoidance = 35   # Prey avoid predators

# Colors for visualization (RGBA tuples, 0-1 range)
prey_color = (0, 1, 0, 1)       # Green for prey
predator_color = (1, 0, 0, 1)   # Red for predators
