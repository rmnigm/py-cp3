import numpy as np
from numba import njit, prange  # type: ignore


def init_boids(boids: np.ndarray, asp: float, v_range: tuple = (0., 1.)) -> np.ndarray:
    """Initialize random boids and their speed in array from uniform distribution"""
    n = boids.shape[0]
    rng = np.random.default_rng()
    low, high = v_range
    boids[:, 0] = rng.uniform(0., asp, size=n)
    boids[:, 1] = rng.uniform(0., 1., size=n)
    alpha = rng.uniform(0, 2*np.pi, size=n)
    v = rng.uniform(low=low, high=high, size=n)
    c, s = np.cos(alpha), np.sin(alpha)
    boids[:, 2] = v * c
    boids[:, 3] = v * s
    return boids


def init_boids_hunters(prey: np.ndarray, predators: np.ndarray, 
                       asp: float, v_range: tuple = (0., 1.)) -> tuple:
    """Initialize random prey and predator boids and their speeds from uniform distribution."""
    # Initialize prey
    n_prey = prey.shape[0]
    rng = np.random.default_rng()
    low, high = v_range
    prey[:, 0] = rng.uniform(0., asp, size=n_prey)
    prey[:, 1] = rng.uniform(0., 1., size=n_prey)
    alpha_prey = rng.uniform(0, 2*np.pi, size=n_prey)
    v_prey = rng.uniform(low=low, high=high, size=n_prey)
    c_prey, s_prey = np.cos(alpha_prey), np.sin(alpha_prey)
    prey[:, 2] = v_prey * c_prey
    prey[:, 3] = v_prey * s_prey
    
    # Initialize predators
    n_pred = predators.shape[0]
    predators[:, 0] = rng.uniform(0., asp, size=n_pred)
    predators[:, 1] = rng.uniform(0., 1., size=n_pred)
    alpha_pred = rng.uniform(0, 2*np.pi, size=n_pred)
    v_pred = rng.uniform(low=low, high=high, size=n_pred)
    c_pred, s_pred = np.cos(alpha_pred), np.sin(alpha_pred)
    predators[:, 2] = v_pred * c_pred
    predators[:, 3] = v_pred * s_pred
    
    return prey, predators


@njit()
def directions(boids: np.ndarray, dt: float) -> np.ndarray:
    """Calculate directions for arrows in boids model by propagating with speed and acceleration"""
    return np.hstack((boids[:, :2] - dt * boids[:, 2:4], boids[:, :2]))


@njit()
def norm(arr: np.ndarray):
    """Calculates norm via first axis"""
    return np.sqrt(np.sum(arr**2, axis=1))


@njit()
def mean_axis(arr, axis):
    """Calculates mean over chosen axis, njit-compilable"""
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = np.mean(arr[:, i])
    else:
        result = np.empty(arr.shape[0], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = np.mean(arr[i, :])
    return result


@njit()
def median_axis(arr, axis):
    """Calculates median over chosen axis, njit-compilable"""
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = np.median(arr[:, i])
    else:
        result = np.empty(arr.shape[0], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = np.median(arr[i, :])
    return result


@njit()
def clip_mag(arr: np.ndarray,
             lims: tuple[float, float] = (0., 1.)):
    """Spots boids with speed greater than limit and clips it to limit"""
    v = norm(arr)
    mask = v > 0
    v_clip = np.clip(v, *lims)
    arr[mask] *= (v_clip[mask] / v[mask]).reshape(-1, 1)


@njit()
def propagate(boids: np.ndarray, dt: float, v_range: tuple):
    """Updates the speed of boids via acceleration, clips it to limit and updates the position of boids"""
    boids[:, 2:4] += dt * boids[:, 4:6]
    clip_mag(boids[:, 2:4], lims=v_range)
    boids[:, 0:2] += dt * boids[:, 2:4]


@njit()
def periodic_walls(boids: np.ndarray, asp: float):
    """Sets the position of boids with respect to periodic walls for them to not fly away"""
    boids[:, 0:2] %= np.array([asp, 1.])


@njit()
def wall_avoidance(boids: np.ndarray, asp: float):
    """Implements wall avoidance component in acceleration logic for boids"""
    left = np.abs(boids[:, 0])
    right = np.abs(asp - boids[:, 0])
    bottom = np.abs(boids[:, 1])
    top = np.abs(1 - boids[:, 1])
    ax = 1. / left**2 - 1. / right**2
    ay = 1. / bottom**2 - 1. / top**2
    boids[:, 4:6] += np.column_stack((ax, ay))


@njit()
def walls(boids: np.ndarray, asp: float):
    """Calculates wall positions with respet to boids position for components in acceleration"""
    c = 1
    x = boids[:, 0]
    y = boids[:, 1]
    a_left = 1 / (np.abs(x) + c)**2
    a_right = -1 / (np.abs(x - asp) + c)**2
    a_bottom = 1 / (np.abs(y) + c)**2
    a_top = -1 / (np.abs(y - 1.) + c)**2
    return np.column_stack((a_left + a_right, a_bottom + a_top))


@njit()
def distance(boids: np.ndarray) -> np.ndarray:
    """Calculates pairwise euclidean distance between boids"""
    p = boids[:, :2]
    n = p.shape[0]
    dist = np.zeros(shape=(n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            v = p[i] - p[j]
            d = (v @ v)
            dist[i, j] = d
    dist = np.sqrt(dist)
    return dist


@njit()
def normalize(v):
    """Normalize vector to norm = 1"""
    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        return v
    return v / v_norm


@njit()
def visibility(boids: np.ndarray, perception: float, angle: float) -> np.ndarray:
    """
    Calculates pairwise euclidean distance between boids, angles and returns mask of visibility,
    implements a sector of angle width = (-arccos(angle), arccos(angle)) and radius = perception
    """
    vectors = boids[:, :2]
    speeds = boids[:, 2:4]
    n = vectors.shape[0]
    dist = np.zeros(shape=(n, n), dtype=np.float64)
    angles = np.zeros(shape=(n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            v = vectors[i] - vectors[j]
            d = (v @ v)
            dist[i, j] = d
            angles[i, j] = np.dot(normalize(speeds[i]), normalize(v))
    dist = np.sqrt(dist)
    distance_mask = dist < perception
    angle_mask = angles > angle
    mask = np.logical_and(distance_mask, angle_mask)
    np.fill_diagonal(mask, False)
    return mask


@njit()
def cohesion(boids: np.ndarray,
             idx: int,
             neigh_mask: np.ndarray,
             perception: float) -> np.ndarray:
    """Implements cohesion component of acceleration via median center of group in sector"""
    center = median_axis(boids[neigh_mask, :2], axis=0)
    a = (center - boids[idx, :2]) / perception
    return a


@njit()
def separation(boids: np.ndarray,
               idx: int,
               neigh_mask: np.ndarray) -> np.ndarray:
    """Implements separation component of acceleration via median within group in sector"""
    d = median_axis(boids[neigh_mask, :2] - boids[idx, :2], axis=0)
    return -d / ((d[0]**2 + d[1]**2) + 1)


@njit()
def alignment(boids: np.ndarray,
              idx: int,
              neigh_mask: np.ndarray,
              v_range: tuple) -> np.ndarray:
    """Implements median-based alingment component of acceleration within group in sector"""
    v_mean = median_axis(boids[neigh_mask, 2:4], axis=0)
    a = (v_mean - boids[idx, 2:4]) / (2 * v_range[1])
    return a


@njit()
def noise():
    """Implements of random noise in (-1, 1) interval for two coordinated, njit-compilable"""
    arr = np.random.rand(2)
    if np.random.rand(1) > .5:
        arr[0] *= -1
    if np.random.rand(1) > .5:
        arr[1] *= -1
    return arr


@njit(parallel=True)
def flocking(boids: np.ndarray,
             perception: float,
             coeffs: np.ndarray,
             asp: float,
             v_range: tuple,
             angle_cos: float = 0.0) -> None:
    """
    Implements boids visibility computation and acceleration computation via four different
    components - cohesion, alignment, separation, noise within sector of certain radius and angle.
    
    Args:
        boids: Array of boids (shape: n x 6)
        perception: Maximum visibility distance
        coeffs: Coefficients for [cohesion, alignment, separation, walls, noise]
        asp: Aspect ratio of simulation area
        v_range: Velocity range tuple (min, max)
        angle_cos: Cosine of half-angle for visibility sector (0.0 = 90°, -1.0 = 180°)
    """
    n = boids.shape[0]
    mask = visibility(boids, perception, angle_cos)
    wal = walls(boids, asp)
    for i in prange(n):
        if not np.any(mask[i]):
            coh = np.zeros(2)
            alg = np.zeros(2)
            sep = np.zeros(2)
            ns = np.zeros(2)
        else:
            coh = cohesion(boids, i, mask[i], perception)
            alg = alignment(boids, i, mask[i], v_range)
            sep = separation(boids, i, mask[i])
            ns = noise()
        boids[i, 4] = (coeffs[0] * coh[0]
                       + coeffs[1] * alg[0]
                       + coeffs[2] * sep[0]
                       + coeffs[3] * wal[i][0]
                       + coeffs[4] * ns[0])
        boids[i, 5] = (coeffs[0] * coh[1]
                       + coeffs[1] * alg[1]
                       + coeffs[2] * sep[1]
                       + coeffs[3] * wal[i][1]
                       + coeffs[4] * ns[0])


def simulation_step(boids: np.ndarray,
                    asp: float,
                    perception: float,
                    coefficients: np.ndarray,
                    v_range: tuple,
                    dt: float,
                    angle_cos: float = 0.0) -> None:
    """Implements full step of boids model simulation with updating their positions and propagation.
    
    Args:
        boids: Array of boids
        asp: Aspect ratio of simulation area
        perception: Maximum visibility distance
        coefficients: Coefficients for flocking behavior
        v_range: Velocity range tuple (min, max)
        dt: Time step
        angle_cos: Cosine of half-angle for visibility sector
    """
    flocking(boids, perception, coefficients, asp, v_range, angle_cos)
    propagate(boids, dt, v_range)
    periodic_walls(boids, asp)
    wall_avoidance(boids, asp)


# +---------------------------------------------------+ #
# Implementation of hunters/prey variant of boids model #
# +---------------------------------------------------+ #

@njit()
def angle_to_cos(angle_deg: float) -> float:
    """Convert angle in degrees to cosine value for visibility calculation.
    
    The angle represents the half-angle of the visibility sector.
    For example, angle_deg=90 means a 180° total sector (front half).
    
    Args:
        angle_deg: Half-angle of the sector in degrees (90-270)
        
    Returns:
        Cosine of the angle for comparison with dot product
    """
    return np.cos(np.radians(angle_deg))


@njit()
def visibility_cross(boids1: np.ndarray, boids2: np.ndarray, 
                     perception: float, angle_cos: float = -1.0) -> np.ndarray:
    """Calculates visibility mask between two different groups of boids.
    
    Visibility is determined by:
    1. Distance within perception radius
    2. Angle within the visibility sector (oriented along velocity vector)
    
    Args:
        boids1: First group of boids (shape: n1 x 6)
        boids2: Second group of boids (shape: n2 x 6)
        perception: Maximum visibility distance
        angle_cos: Cosine of half-angle for visibility sector (-1.0 = 360° visibility)
        
    Returns:
        Boolean mask of shape (n1, n2) indicating which boids2 are visible to each boids1
    """
    p1 = boids1[:, :2]
    p2 = boids2[:, :2]
    s1 = boids1[:, 2:4]  # velocities of boids1
    n1 = p1.shape[0]
    n2 = p2.shape[0]
    mask = np.zeros((n1, n2), dtype=np.bool_)
    
    for i in range(n1):
        # Calculate normalized velocity direction for boids1[i]
        speed_norm = np.sqrt(s1[i, 0]**2 + s1[i, 1]**2)
        if speed_norm > 0:
            speed_dir = s1[i] / speed_norm
        else:
            speed_dir = np.array([0.0, 0.0])
        
        for j in range(n2):
            # Direction FROM boids1[i] TO boids2[j]
            v = p2[j] - p1[i]
            d = np.sqrt(v[0]**2 + v[1]**2)
            
            if d < perception and d > 0:
                if angle_cos <= -0.9999:  # 360° visibility (angle >= 180°)
                    mask[i, j] = True
                else:
                    # Check if within visibility sector
                    v_norm = v / d
                    dot = speed_dir[0] * v_norm[0] + speed_dir[1] * v_norm[1]
                    if dot > angle_cos:
                        mask[i, j] = True
    return mask


@njit()
def separation_cross(boids1: np.ndarray, idx: int, boids2: np.ndarray,
                     cross_mask: np.ndarray) -> np.ndarray:
    """Calculates separation acceleration between boids of different groups"""
    if not np.any(cross_mask):
        return np.zeros(2)
    
    # Calculate median position of visible boids from other group
    visible_pos = boids2[cross_mask, :2]
    d = median_axis(visible_pos - boids1[idx, :2], axis=0)
    return -d / ((d[0]**2 + d[1]**2) + 1)


@njit(parallel=True)
def flocking_hunters(prey: np.ndarray, predators: np.ndarray,
                     perception: float,
                     prey_coeffs: np.ndarray, pred_coeffs: np.ndarray,
                     pred_to_prey_attraction: float, prey_to_predator_avoidance: float,
                     asp: float, v_range: tuple,
                     angle_cos: float = 0.0,
                     angle_prey_pred_cos: float = -1.0,
                     angle_pred_prey_cos: float = -1.0) -> None:
    """Implements flocking behavior for both prey and predator boids with cross-species interactions.
    
    Uses sector-based visibility where agents can only see others within a cone oriented
    along their velocity vector.
    
    Args:
        prey: Array of prey boids (shape: n_prey x 6)
        predators: Array of predator boids (shape: n_pred x 6)
        perception: Maximum visibility distance
        prey_coeffs: Coefficients for prey flocking behavior
        pred_coeffs: Coefficients for predator flocking behavior
        pred_to_prey_attraction: Attraction strength of predators to prey
        prey_to_predator_avoidance: Avoidance strength of prey from predators
        asp: Aspect ratio of the simulation area
        v_range: Velocity range tuple (min, max)
        angle_cos: Cosine of half-angle for same-species visibility sector
        angle_prey_pred_cos: Cosine of half-angle for prey seeing predators
        angle_pred_prey_cos: Cosine of half-angle for predators seeing prey
    """
    n_prey = prey.shape[0]
    n_pred = predators.shape[0]
    
    # Visibility within prey group (same-species angle)
    mask_prey = visibility(prey, perception, angle_cos)
    # Visibility within predator group (same-species angle)
    mask_pred = visibility(predators, perception, angle_cos)
    # Cross-visibility: predators seeing prey
    pred_see_prey = visibility_cross(predators, prey, perception, angle_pred_prey_cos)
    # Cross-visibility: prey seeing predators
    prey_see_pred = visibility_cross(prey, predators, perception, angle_prey_pred_cos)
    
    # Wall forces
    wal_prey = walls(prey, asp)
    wal_pred = walls(predators, asp)
    
    # Update prey
    for i in prange(n_prey):
        # Standard flocking with other prey
        if not np.any(mask_prey[i]):
            coh = np.zeros(2)
            alg = np.zeros(2)
            sep_prey = np.zeros(2)
        else:
            coh = cohesion(prey, i, mask_prey[i], perception)
            alg = alignment(prey, i, mask_prey[i], v_range)
            sep_prey = separation(prey, i, mask_prey[i])
        
        # Separation from predators (increased - avoidance)
        sep_pred = separation_cross(prey, i, predators, prey_see_pred[i])
        
        ns = noise()
        
        prey[i, 4] = (prey_coeffs[0] * coh[0]
                      + prey_coeffs[1] * alg[0]
                      + prey_coeffs[2] * sep_prey[0]
                      + prey_coeffs[3] * wal_prey[i][0]
                      + prey_coeffs[4] * ns[0]
                      + prey_to_predator_avoidance * sep_pred[0])
        prey[i, 5] = (prey_coeffs[0] * coh[1]
                      + prey_coeffs[1] * alg[1]
                      + prey_coeffs[2] * sep_prey[1]
                      + prey_coeffs[3] * wal_prey[i][1]
                      + prey_coeffs[4] * ns[1]
                      + prey_to_predator_avoidance * sep_pred[1])
    
    # Update predators
    for i in prange(n_pred):
        # Reduced flocking with other predators
        if not np.any(mask_pred[i]):
            coh = np.zeros(2)
            alg = np.zeros(2)
            sep_pred = np.zeros(2)
        else:
            coh = cohesion(predators, i, mask_pred[i], perception)
            alg = alignment(predators, i, mask_pred[i], v_range)
            sep_pred = separation(predators, i, mask_pred[i])
        
        # Separation from prey (reduced - attraction/chase)
        sep_prey = separation_cross(predators, i, prey, pred_see_prey[i])
        
        ns = noise()
        
        predators[i, 4] = (pred_coeffs[0] * coh[0]
                           + pred_coeffs[1] * alg[0]
                           + pred_coeffs[2] * sep_pred[0]
                           + pred_coeffs[3] * wal_pred[i][0]
                           + pred_coeffs[4] * ns[0]
                           + pred_to_prey_attraction * sep_prey[0])
        predators[i, 5] = (pred_coeffs[0] * coh[1]
                           + pred_coeffs[1] * alg[1]
                           + pred_coeffs[2] * sep_pred[1]
                           + pred_coeffs[3] * wal_pred[i][1]
                           + pred_coeffs[4] * ns[1]
                           + pred_to_prey_attraction * sep_prey[1])


def simulation_step_hunters(prey: np.ndarray, predators: np.ndarray,
                            asp: float, perception: float,
                            prey_coeffs: np.ndarray, pred_coeffs: np.ndarray,
                            pred_to_prey_attraction: float, prey_to_predator_avoidance: float,
                            v_range: tuple, dt: float,
                            angle_cos: float = 0.0,
                            angle_prey_pred_cos: float = -1.0,
                            angle_pred_prey_cos: float = -1.0) -> None:
    """Implements full step of hunters/prey simulation with updating positions and propagation.
    
    Args:
        prey: Array of prey boids
        predators: Array of predator boids
        asp: Aspect ratio of simulation area
        perception: Maximum visibility distance
        prey_coeffs: Coefficients for prey flocking behavior
        pred_coeffs: Coefficients for predator flocking behavior
        pred_to_prey_attraction: Attraction strength of predators to prey
        prey_to_predator_avoidance: Avoidance strength of prey from predators
        v_range: Velocity range tuple (min, max)
        dt: Time step
        angle_cos: Cosine of half-angle for same-species visibility
        angle_prey_pred_cos: Cosine of half-angle for prey seeing predators
        angle_pred_prey_cos: Cosine of half-angle for predators seeing prey
    """
    flocking_hunters(prey, predators, perception,
                     prey_coeffs, pred_coeffs,
                     pred_to_prey_attraction, prey_to_predator_avoidance,
                     asp, v_range, angle_cos, angle_prey_pred_cos, angle_pred_prey_cos)
    propagate(prey, dt, v_range)
    propagate(predators, dt, v_range)
    periodic_walls(prey, asp)
    periodic_walls(predators, asp)
    wall_avoidance(prey, asp)
    wall_avoidance(predators, asp)
