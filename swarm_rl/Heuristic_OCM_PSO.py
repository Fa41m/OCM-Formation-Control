import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------------------------
# Global Parameters
# ---------------------------------------------
formation_type = "circle"  # "circle" or "triangle"

num_robots = 15
num_steps = 800

robot_max_speed = 0.3
robot_radius = 0.05
robot_diameter = 2 * robot_radius
robot_collision_zone = (2 * robot_radius)

# Robot Sensor Parameters
sensor_detection_distance = robot_diameter * 40
sensor_buffer_radius = robot_diameter * 10
object_sensor_buffer_radius = robot_diameter * 5

# alpha_base = 0.05    # repulsion for robots to avoid collisions
# beta_base = 0.77     # repulsion from obstacles
alpha_base = 0.4    # repulsion for robots to avoid collisions
beta_base = 0.4     # repulsion from obstacles

formation_radius_base = num_robots // 5
formation_size_triangle_base = num_robots // 10

K_base = 0.8  # Base alignment strength
C_base = 0.7  # Base cohesion strength
# K_base = 0.08  # Base alignment strength
# C_base = 0.41  # Base cohesion strength
K_min, K_max = 0.005, 0.995
C_min, C_max = 0.005, 0.995

K_values_over_time = []
C_values_over_time = []
alpha_values_over_time = []
beta_values_over_time = []

lambda_K = 0.7
lambda_C = 0.5

world_width = num_robots * 4
world_boundary_tolerance = robot_diameter * 5

# Obstacle Parameters
obstacle_level = 1
num_obstacles = 3
min_obstacle_size = world_width / 50
max_obstacle_size = world_width / 25
offset_degrees = 50
passage_width = world_width / 15

# Circle path
circle_center = np.array([world_width / 2, world_width / 2])
circle_radius = world_width / 4

# ------------------------------------------------------
# New Cost Function Weights (instead of old cost_w1, etc.)
# ------------------------------------------------------
cost_w_path  = 0.02   # weight for path-tracking error
cost_w_obs   = 1.0    # weight for obstacle proximity penalty
cost_w_align = 0.05   # weight for swarm alignment
cost_w_force = 0.005  # weight for control effort

# Track total cost over the entire simulation
total_cost = 0.0

# ---------------------------------------------
# Helper Functions
# ---------------------------------------------
def generate_varied_obstacles_with_levels(center, radius, num_obstacles,
                                          min_size, max_size, offset_degrees,
                                          passage_width, level):
    offset_radians = np.deg2rad(offset_degrees)
    angles = np.linspace(0, 2 * np.pi, num_obstacles, endpoint=False) + offset_radians
    obstacles = []
    for i, angle in enumerate(angles):
        if level == 1 or (level == 4 and i % 3 == 0):  # Offset obstacles
            offset_distance = np.random.choice([-1, 1]) * (passage_width - 1)
            pos = center + (radius + offset_distance) * np.array([np.cos(angle), np.sin(angle)])
            size = np.random.uniform(min_size, max_size)
            obstacles.append({"type": "circle", "position": pos, "radius": size})

        elif level == 2 or (level == 4 and i % 3 == 1):  # Obstacles on the circle
            pos = center + radius * np.array([np.cos(angle), np.sin(angle)])
            size = np.random.uniform(min_size, max_size)
            obstacles.append({"type": "circle", "position": pos, "radius": size})

        elif level == 3 or (level == 4 and i % 3 == 2):  # Paired obstacles with a passage
            central_pos = center + radius * np.array([np.cos(angle), np.sin(angle)])
            size1 = np.random.uniform(min_size, max_size)
            size2 = np.random.uniform(min_size, max_size)
            adjusted_width = max(passage_width, size1 + size2 + passage_width)
            pos1 = central_pos - adjusted_width / 2 * np.array([np.cos(angle), np.sin(angle)])
            pos2 = central_pos + adjusted_width / 2 * np.array([np.cos(angle), np.sin(angle)])
            obstacles.extend([
                {"type": "circle", "position": pos1, "radius": size1},
                {"type": "circle", "position": pos2, "radius": size2}
            ])
    return obstacles

# ---------------------------------------------
# Circle Formation
# ---------------------------------------------
def initialize_positions(num_robots, start_position, formation_radius):
    angles = np.linspace(0, 2 * np.pi, num_robots, endpoint=False)
    return np.array([
        start_position + formation_radius * np.array([np.cos(angle), np.sin(angle)])
        for angle in angles
    ])

moving_theta = 0
theta_step = 2 * np.pi / num_steps

def get_moving_center(frame, total_frames, swarm_positions):
    global circle_center, circle_radius, moving_theta

    swarm_center = np.mean(swarm_positions, axis=0)
    expected_position = circle_center + circle_radius * np.array([np.cos(moving_theta), np.sin(moving_theta)])
    lag_distance = np.linalg.norm(swarm_center - expected_position)

    lag_threshold = 5.0
    min_slow_factor = 0.05
    max_lag_distance = 12.0

    if lag_distance > max_lag_distance:
        slow_factor = 0.0
    else:
        slow_factor = max(min_slow_factor, min(1.0, lag_threshold / (lag_distance + 1e-6)))

    moving_theta += theta_step * slow_factor
    adjusted_center = circle_center + circle_radius * np.array([np.cos(moving_theta), np.sin(moving_theta)])
    return adjusted_center

def get_target_positions(moving_center, num_robots, formation_radius):
    angles = np.linspace(0, 2 * np.pi, num_robots, endpoint=False)
    return np.array([
        moving_center + formation_radius * np.array([np.cos(angle), np.sin(angle)])
        for angle in angles
    ])

# ---------------------------------------------
# Triangle Formation
# ---------------------------------------------
def initialize_positions_triangle(num_robots, start_position, formation_size):
    radius_vector = start_position - circle_center
    heading_angle = np.arctan2(radius_vector[1], radius_vector[0])

    row = 1
    total_bots = 0
    rows = []
    while total_bots < num_robots:
        bots_in_row = row
        if total_bots + bots_in_row > num_robots:
            bots_in_row = num_robots - total_bots
        rows.append(bots_in_row)
        total_bots += bots_in_row
        row += 1

    positions_relative = []
    h_spacing = formation_size * 1.5
    v_spacing = formation_size
    y_offset = 0
    for bots_in_row in rows:
        x_offset = -v_spacing * (bots_in_row - 1) / 2.0
        for j in range(bots_in_row):
            positions_relative.append([x_offset + j * v_spacing, y_offset])
        y_offset -= h_spacing

    positions_relative = np.array(positions_relative)
    rotation_matrix = np.array([
        [np.cos(heading_angle), -np.sin(heading_angle)],
        [np.sin(heading_angle),  np.cos(heading_angle)]
    ])
    rotated_positions = positions_relative @ rotation_matrix.T
    return rotated_positions + start_position

def get_target_positions_triangle(moving_center, num_robots, formation_size):
    radius_vector = moving_center - circle_center
    heading_angle = np.arctan2(radius_vector[1], radius_vector[0])

    row = 1
    total_bots = 0
    rows = []
    while total_bots < num_robots:
        bots_in_row = row
        if total_bots + bots_in_row > num_robots:
            bots_in_row = num_robots - total_bots
        rows.append(bots_in_row)
        total_bots += bots_in_row
        row += 1

    positions_relative = []
    h_spacing = formation_size * 1.5
    v_spacing = formation_size
    y_offset = 0
    for bots_in_row in rows:
        x_offset = -v_spacing * (bots_in_row - 1) / 2.0
        for j in range(bots_in_row):
            positions_relative.append([x_offset + j * v_spacing, y_offset])
        y_offset -= h_spacing

    positions_relative = np.array(positions_relative)
    rotation_matrix = np.array([
        [np.cos(heading_angle), -np.sin(heading_angle)],
        [np.sin(heading_angle),  np.cos(heading_angle)]
    ])
    rotated_positions = positions_relative @ rotation_matrix.T
    return rotated_positions + moving_center

# ---------------------------------------------
# Sensor / Avoidance
# ---------------------------------------------
def raycast_sensor(position, heading, sensor_angle, obstacles, sensor_detection_distance):
    sensor_direction = np.array([np.cos(heading + sensor_angle), np.sin(heading + sensor_angle)])
    sensor_start = position
    min_distance = sensor_detection_distance
    repulsion_vector = np.zeros(2)

    for obs in obstacles:
        obs_pos = obs["position"]
        obs_radius = obs["radius"]

        to_obstacle = obs_pos - sensor_start
        projection_length = np.dot(to_obstacle, sensor_direction)

        if 0 < projection_length < sensor_detection_distance:
            closest_point = sensor_start + projection_length * sensor_direction
            distance_to_obstacle = np.linalg.norm(closest_point - obs_pos)
            if distance_to_obstacle <= obs_radius + object_sensor_buffer_radius:
                intersection_distance = (
                    projection_length -
                    np.sqrt((obs_radius + object_sensor_buffer_radius)**2 - distance_to_obstacle**2)
                )
                if 0 < intersection_distance < min_distance:
                    min_distance = intersection_distance
                    repulsion_vector = sensor_start - obs_pos
                    denom = np.linalg.norm(repulsion_vector) + 1e-6
                    repulsion_vector /= denom

    return min_distance, repulsion_vector

def smooth_transition_with_bounds(current_value, target_value, smoothing_factor, value_min, value_max):
    new_value = current_value + smoothing_factor * (target_value - current_value)
    new_value = np.clip(new_value, value_min, value_max)
    if abs(new_value - target_value) < 0.005:
        new_value = target_value
    return new_value

def adjust_alignment_cohesion_gradual(current_K, current_C, center_dist):
    if center_dist < sensor_detection_distance:
        target_K = max(K_min, current_K * (center_dist / sensor_detection_distance))
        target_C = max(C_min, current_C * (center_dist / sensor_detection_distance))
    else:
        target_K = min(K_base, current_K * 1.05)
        target_C = min(C_base, current_C * 1.05)

    new_K = smooth_transition_with_bounds(current_K, target_K, lambda_K, K_min, K_max)
    new_C = smooth_transition_with_bounds(current_C, target_C, lambda_C, C_min, C_max)
    return new_K, new_C

# ---------------------------------------------
# Adaptation of formation, alpha, beta
# ---------------------------------------------
def adapt_parameters(positions, obstacles, base_formation_radius, base_formation_size_triangle,
                     base_alpha, base_beta, min_dist_threshold=4.0):
    if len(positions) == 0:
        return (base_formation_radius, base_formation_size_triangle, base_alpha, base_beta)

    min_obstacle_dist = float('inf')
    obstacle_radius_for_closest = 0.0
    for obs in obstacles:
        obs_center = obs["position"]
        obs_radius = obs["radius"]
        for pos in positions:
            dist = np.linalg.norm(pos - obs_center) - obs_radius
            if dist < min_obstacle_dist:
                min_obstacle_dist = dist
                obstacle_radius_for_closest = obs_radius

    min_robot_dist = float('inf')
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_robot_dist:
                min_robot_dist = d

    obs_factor = 0.0
    if min_obstacle_dist < min_dist_threshold:
        obs_factor = (min_dist_threshold - min_obstacle_dist) / min_dist_threshold
        obs_factor = np.clip(obs_factor, 0.0, 1.0)

    large_obstacle_threshold = 3.0
    if obstacle_radius_for_closest > large_obstacle_threshold:
        obs_factor = min(obs_factor * 1.2, 1.0)

    shrink_ratio = 1.0 - 0.5 * obs_factor
    new_formation_radius = base_formation_radius * shrink_ratio
    new_formation_size_triangle = base_formation_size_triangle * shrink_ratio

    density_factor = 0.0
    density_threshold = 1.0
    if min_robot_dist < density_threshold:
        density_factor = (density_threshold - min_robot_dist) / density_threshold
        density_factor = np.clip(density_factor, 0.0, 1.0)
    new_alpha = base_alpha * (1.0 + 0.5 * density_factor)

    if obs_factor > 0.1:
        new_beta = min(base_beta * (1.5 * obs_factor), base_beta * 2)
    else:
        new_beta = max(base_beta * 0.8, base_beta)

    return (new_formation_radius, new_formation_size_triangle, new_alpha, new_beta)

# ---------------------------------------------
# Force Computation (with 3 sensors & side logic)
# ---------------------------------------------
def compute_forces_with_sensors(positions, headings, velocities, target_positions,
                                obstacles, current_K, current_C, alpha, beta):
    num_robots = len(positions)
    forces = np.zeros((num_robots, 2))

    left_angle = np.radians(30)
    center_angle = 0.0
    right_angle = np.radians(-30)

    for i in range(num_robots):
        alignment_force = np.zeros(2)
        cohesion_force = target_positions[i] - positions[i]
        robot_repulsion_force = np.zeros(2)

        heading_dir = np.array([np.cos(headings[i]), np.sin(headings[i])])
        left_dir   = np.array([-heading_dir[1],  heading_dir[0]])
        right_dir  = np.array([ heading_dir[1], -heading_dir[0]])

        dist_left, _   = raycast_sensor(positions[i], headings[i], left_angle, obstacles, sensor_detection_distance)
        dist_center, _ = raycast_sensor(positions[i], headings[i], center_angle, obstacles, sensor_detection_distance)
        dist_right, _  = raycast_sensor(positions[i], headings[i], right_angle, obstacles, sensor_detection_distance)

        # min distance used to adapt K & C
        min_distance = min(dist_left, dist_center, dist_right)
        updated_K, updated_C = adjust_alignment_cohesion_gradual(current_K, current_C, min_distance)
        current_K = updated_K
        current_C = updated_C

        def avoidance_scale(d):
            return beta * max(0.0, (sensor_detection_distance - d)) / sensor_detection_distance

        avoidance_force = np.zeros(2)
        if dist_left < sensor_detection_distance:
            avoidance_force += avoidance_scale(dist_left) * right_dir
        if dist_right < sensor_detection_distance:
            avoidance_force += avoidance_scale(dist_right) * left_dir
        if dist_center < sensor_detection_distance:
            if dist_left > dist_right:
                avoidance_force += avoidance_scale(dist_center) * left_dir
            else:
                avoidance_force += avoidance_scale(dist_center) * right_dir

        # Robot-Robot Repulsion
        for j in range(num_robots):
            if i == j:
                continue
            direction = positions[i] - positions[j]
            distance = np.linalg.norm(direction)
            if distance < sensor_buffer_radius:
                effective_distance = max(distance, 1e-4)
                robot_repulsion_force += (direction / effective_distance) * (alpha / effective_distance)

        # Alignment
        neighbors = [
            j for j in range(num_robots)
            if i != j and np.linalg.norm(positions[i] - positions[j]) < sensor_detection_distance
        ]
        if neighbors:
            avg_heading = np.mean([headings[n] for n in neighbors])
            alignment_force = np.array([
                np.cos(avg_heading) - np.cos(headings[i]),
                np.sin(avg_heading) - np.sin(headings[i])
            ])

        # Combine Forces
        forces[i] = (updated_C * cohesion_force
                     + avoidance_force
                     + updated_K * alignment_force
                     + robot_repulsion_force)

    return forces, current_K, current_C

# ---------------------------------------------
# Collision Checking
# ---------------------------------------------
def check_collisions(positions, obstacles):
    to_remove = set()
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < robot_collision_zone:
                to_remove.add(i)
                to_remove.add(j)

    for i, pos in enumerate(positions):
        for obs in obstacles:
            obs_pos = obs["position"]
            obs_radius = obs["radius"]
            if np.linalg.norm(pos - obs_pos) < (obs_radius + robot_collision_zone):
                to_remove.add(i)

    to_remove_list = sorted(to_remove)
    updated_positions = np.delete(positions, to_remove_list, axis=0)
    return updated_positions, to_remove_list

def enforce_boundary_conditions(positions, world_width, world_boundary_tolerance):
    return np.clip(positions, world_boundary_tolerance, world_width - world_boundary_tolerance)

def update_positions_and_headings(positions, headings, forces, robot_max_speed, boundary_conditions):
    for i in range(len(positions)):
        velocity = forces[i]
        speed = np.linalg.norm(velocity)
        if speed > robot_max_speed:
            velocity = (robot_max_speed / speed) * velocity
        positions[i] += velocity
        if np.linalg.norm(velocity) > 1e-6:
            headings[i] = np.arctan2(velocity[1], velocity[0])
    positions = enforce_boundary_conditions(positions, *boundary_conditions)
    return positions, headings

# -------------------------------------------
# Alignment Utility
# -------------------------------------------
def compute_swarm_alignment(headings):
    """Returns psi in [0..1], with 1 = perfectly aligned."""
    avg_cos = np.mean(np.cos(headings))
    avg_sin = np.mean(np.sin(headings))
    psi = np.sqrt(avg_cos**2 + avg_sin**2)
    return psi

# -------------------------------------------
# Main Function
# -------------------------------------------
def run_simulation(alpha, beta, K, C, do_plot=False):
    """
    Run the swarm simulation using the provided alpha, beta, K, and C values.

    Parameters
    ----------
    alpha : float
        Value to override the global alpha_base.
    beta : float
        Value to override the global beta_base.
    K : float
        Value to override the global K_base (alignment).
    C : float
        Value to override the global C_base (cohesion).
    do_plot : bool
        Whether to do any post-simulation plotting (optional).

    Returns
    -------
    total_cost : float
        Final accumulated cost of the simulation.
    """
    # 1) Override global parameters with the values passed in
    global alpha_base, beta_base, K_base, C_base
    alpha_base = alpha
    beta_base  = beta
    K_base     = K
    C_base     = C

    # 2) Reset cost tracking
    global total_cost, t_rise, t_rise_recorded
    total_cost = 0.0
    t_rise = num_steps
    t_rise_recorded = False

    # 3) Initialize the swarm (same logic as your main, but no plotting yet)
    start_position = circle_center + circle_radius * np.array([1, 0])
    headings = np.random.uniform(0, 2 * np.pi, num_robots)
    velocities = np.zeros((num_robots, 2))

    if formation_type.lower() == "triangle":
        positions = initialize_positions_triangle(num_robots, start_position, num_robots // 10)
        get_target_positions_fn = get_target_positions_triangle
    else:
        positions = initialize_positions(num_robots, start_position, num_robots // 5)
        get_target_positions_fn = get_target_positions

    # 4) Create obstacles
    obstacles = generate_varied_obstacles_with_levels(
        circle_center,
        circle_radius,
        3,   # num_obstacles
        world_width / 50,
        world_width / 25,
        50,  # offset_degrees
        world_width / 15,
        4    # obstacle_level
    )

    current_K = K_base
    current_C = C_base
    formation_radius_base = num_robots // 5
    formation_size_triangle_base = num_robots // 10
    formation_radius = formation_radius_base
    formation_size_triangle = formation_size_triangle_base

    # For optional plotting or diagnostics:
    K_values_over_time = [current_K]
    C_values_over_time = [current_C]
    alpha_values_over_time = [alpha_base]
    beta_values_over_time = [beta_base]
    cost_history = [0.0]

    # 5) Main simulation loop
    for frame in range(num_steps):
        # A) Collisions
        updated_positions, removed_indices = check_collisions(positions, obstacles)
        if removed_indices:
            headings = np.delete(headings, removed_indices, axis=0)
            velocities = np.delete(velocities, removed_indices, axis=0)
            positions = updated_positions
        else:
            positions = updated_positions

        if len(positions) == 0:
            # All robots removed => cost might be very large or just break
            # Let's break here for demonstration:
            break

        # B) Adapt parameters (if that's part of your logic)
        formation_radius, formation_size_triangle, alpha_curr, beta_curr = adapt_parameters(
            positions, obstacles,
            formation_radius_base, formation_size_triangle_base,
            alpha_base, beta_base
        )

        # C) Generate target positions
        moving_center = get_moving_center(frame, num_steps, positions)
        if formation_type.lower() == "triangle":
            target_positions = get_target_positions_triangle(moving_center, len(positions), formation_size_triangle)
        else:
            target_positions = get_target_positions(moving_center, len(positions), formation_radius)

        # D) Compute forces & update K, C
        forces, current_K, current_C = compute_forces_with_sensors(
            positions,
            headings,
            velocities,
            target_positions,
            obstacles,
            current_K,
            current_C,
            alpha_curr,
            beta_curr
        )

        # E) Update positions & headings
        positions, headings = update_positions_and_headings(
            positions,
            headings,
            forces,
            robot_max_speed,
            (world_width, world_boundary_tolerance)
        )

        # ------------------------------------------------
        # 8) NEW COST FUNCTION: path + collision + alignment + control effort
        # ------------------------------------------------

        # 8a) Swarm alignment cost
        psi = compute_swarm_alignment(headings)
        alignment_cost = cost_w_align * (1.0 - psi)**2  # 0 when perfectly aligned

        # 8b) Path-following error: distance from each robot to its target position
        path_errors = np.linalg.norm(positions - target_positions, axis=1)
        avg_path_error = np.mean(path_errors) if len(path_errors) > 0 else 0.0
        path_cost = cost_w_path * avg_path_error

        # 8c) Collision-avoidance cost (penalize robots that get too close to obstacles)
        #     We'll sum an exponential penalty for each robot that's near an obstacle.
        collision_cost = 0.0
        safe_distance = sensor_detection_distance / 2.0  # or pick some threshold
        for i in range(len(positions)):
            for obs in obstacles:
                obs_dist = np.linalg.norm(positions[i] - obs["position"]) - obs["radius"]
                if obs_dist < safe_distance:
                    # exponential penalty increases as the robot gets closer
                    collision_cost += cost_w_obs * np.exp(-obs_dist)

        # 8d) Control effort: sum of the magnitudes of the forces
        sum_force = np.sum(np.linalg.norm(forces, axis=1))
        control_cost = cost_w_force * sum_force

        # Combine all cost terms for this timestep
        cost_step = alignment_cost + path_cost + collision_cost + control_cost
        total_cost += cost_step

        # I) Track for plotting if needed
        K_values_over_time.append(current_K)
        C_values_over_time.append(current_C)
        alpha_values_over_time.append(alpha_curr)
        beta_values_over_time.append(beta_curr)
        cost_history.append(total_cost)

    # Optional: Plot if requested
    if do_plot:
        # Plot K and C
        plt.figure(figsize=(10, 6))
        plt.plot(K_values_over_time, label='Alignment (K)', color='blue')
        plt.plot(C_values_over_time, label='Cohesion (C)', color='green')
        plt.xlabel('Time Step')
        plt.ylabel('Values')
        plt.title('Alignment (K) and Cohesion (C) Over Time')
        plt.legend()
        plt.show()

        # Plot alpha and beta
        plt.figure(figsize=(10, 6))
        plt.plot(alpha_values_over_time, label='Alpha', color='blue')
        plt.plot(beta_values_over_time, label='Beta', color='green')
        plt.xlabel('Time Step')
        plt.ylabel('Values')
        plt.title('Alpha and Beta Over Time')
        plt.legend()
        plt.show()

        # Plot cost evolution
        plt.figure(figsize=(10, 6))
        plt.plot(cost_history, label='Accumulated Cost', color='red')
        plt.xlabel('Time Step')
        plt.ylabel('Accumulated Cost')
        plt.title('Cost Function Evolution Over Time')
        plt.legend()
        plt.show()

    # Return the final accumulated cost
    return total_cost

def pso_optimize(
    n_particles=20,      # Increased number of particles for better exploration
    n_iterations=4,     # Increased number of iterations for refined solutions
    alpha_bounds=(0.1, 1.0),
    beta_bounds=(0.1, 1.0),
    K_bounds=(0.1, 0.99),
    C_bounds=(0.1, 0.99),
    w_max=1.2,            # Adaptive inertia weight (starting value)
    w_min=0.4,            # Adaptive inertia weight (final value)
    c1=2.05,              # Cognitive component
    c2=2.05,              # Social component
    cv_threshold=0.05,    # Coefficient of variation threshold for early stopping
    cv_window=20          # Window size to check variation in cost values
):
    """
    Optimizes alpha, beta, K, and C for the OCM algorithm using Particle Swarm Optimization (PSO)
    with adaptive inertia weight, larger population, and coefficient of variation-based termination.

    Returns:
    - best_params: Optimal values of (alpha, beta, K, C)
    - best_cost: Corresponding minimized cost function value
    """

    # Initialize swarm positions and velocities
    swarm_pos = np.random.uniform(
        [alpha_bounds[0], beta_bounds[0], K_bounds[0], C_bounds[0]],
        [alpha_bounds[1], beta_bounds[1], K_bounds[1], C_bounds[1]],
        (n_particles, 4)
    )
    swarm_vel = np.random.uniform(-0.01, 0.01, (n_particles, 4))

    # Initialize personal best positions and costs
    personal_best_positions = swarm_pos.copy()
    personal_best_costs = np.array([run_simulation(*swarm_pos[i]) for i in range(n_particles)])

    # Global best
    global_best_idx = np.argmin(personal_best_costs)
    global_best_position = personal_best_positions[global_best_idx].copy()
    global_best_cost = personal_best_costs[global_best_idx]

    # Store recent best costs for CV termination
    recent_best_costs = []

    for iteration in range(n_iterations):
        # Adapt inertia weight over time
        w = w_max - ((w_max - w_min) * (iteration / n_iterations))

        for i in range(n_particles):
            r1 = np.random.rand(4)
            r2 = np.random.rand(4)

            # Velocity update with constriction factor
            phi = c1 + c2
            k = 2 / abs(2 - phi - np.sqrt(phi ** 2 - 4 * phi))  # Constriction coefficient

            swarm_vel[i] = (
                k * (w * swarm_vel[i] 
                     + c1 * r1 * (personal_best_positions[i] - swarm_pos[i])
                     + c2 * r2 * (global_best_position - swarm_pos[i]))
            )

            # Update position and enforce boundary constraints
            swarm_pos[i] += swarm_vel[i]
            swarm_pos[i] = np.clip(swarm_pos[i], 
                                   [alpha_bounds[0], beta_bounds[0], K_bounds[0], C_bounds[0]], 
                                   [alpha_bounds[1], beta_bounds[1], K_bounds[1], C_bounds[1]])

            # Evaluate new cost (Monte Carlo Averaging: Run multiple simulations per particle)
            num_mc_runs = 5  # Increase for more robustness
            costs = [run_simulation(*swarm_pos[i]) for _ in range(num_mc_runs)]
            avg_cost = np.mean(costs)

            # Update personal best
            if avg_cost < personal_best_costs[i]:
                personal_best_costs[i] = avg_cost
                personal_best_positions[i] = swarm_pos[i].copy()

                # Update global best if necessary
                if avg_cost < global_best_cost:
                    global_best_cost = avg_cost
                    global_best_position = swarm_pos[i].copy()

        # Store the best cost in recent history
        recent_best_costs.append(global_best_cost)
        if len(recent_best_costs) > cv_window:
            recent_best_costs.pop(0)

        # Early stopping: Check Coefficient of Variation (CV)
        if len(recent_best_costs) == cv_window:
            cost_std = np.std(recent_best_costs)
            cost_mean = np.mean(recent_best_costs)
            cv = cost_std / (cost_mean + 1e-6)  # Avoid division by zero

            if cv < cv_threshold:
                print(f"Convergence reached at iteration {iteration+1} with CV={cv:.5f}")
                break

        print(f"Iteration {iteration+1}/{n_iterations}, Best Cost: {global_best_cost:.4f}")

    return global_best_position, global_best_cost

# ----------------------------------------------------------
# Example main function that calls run_simulation
# ----------------------------------------------------------
def main():
    print("Running PSO with improved optimization...")

    best_params, best_cost = pso_optimize(
        n_particles=10,
        n_iterations=10,
        alpha_bounds=(0.001, 1.0),
        beta_bounds=(0.001, 1.0),
        K_bounds=(0.005, 0.995),
        C_bounds=(0.005, 0.995),
        w_max=1.2,
        w_min=0.4,
        c1=2.05,
        c2=2.05,
        cv_threshold=0.05,
        cv_window=20
    )

    print("\nPSO Optimization Completed!")
    print(f"Best Parameters (α, β, K, C): {best_params}")
    print(f"Best Cost: {best_cost:.4f}")
    
    # Run a final simulation with the best parameters
    run_simulation(*best_params, do_plot=True)

if __name__ == "__main__":
    main()