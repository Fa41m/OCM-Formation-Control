import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------------------------
# Global Parameters
# ---------------------------------------------
formation_type = "circle"  # "circle" or "triangle"

num_robots = 15
num_steps = 400

# Robot radius in meters
robot_max_speed = 0.3
robot_radius = 0.05
robot_diameter = 2 * robot_radius
robot_collision_zone = (2 * robot_radius)

# Robot Sensor Parameters
sensor_detection_distance = robot_diameter * 40
sensor_buffer_radius = robot_diameter * 10
object_sensor_buffer_radius = robot_diameter * 5

# Baseline repulsion strengths
alpha_base = 0.4    # repulsion for robots to avoid collisions
beta_base = 0.4     # repulsion from obstacles

# For circle formation
formation_radius_base = num_robots // 5

# For triangle formation
formation_size_triangle_base = num_robots // 10

# We'll keep K_base and C_base for alignment & cohesion
K_base = 0.8  # Base alignment strength
C_base = 0.7  # Base cohesion strength
K_min, K_max = 0.005, 0.995  # Range for alignment strength
C_min, C_max = 0.005, 0.995  # Range for cohesion strength

# Arrays to store K and C values over time
K_values_over_time = []
C_values_over_time = []
alpha_values_over_time = []
beta_values_over_time = []

# Smoothing factors
lambda_K = 0.7
lambda_C = 0.5

# World
world_width = num_robots * 4
world_boundary_tolerance = robot_diameter * 5

# Obstacle Parameters
obstacle_level = 4
num_obstacles = 3
min_obstacle_size = world_width / 50
max_obstacle_size = world_width / 25
offset_degrees = 50
passage_width = world_width / 15

# Circle center
circle_center = np.array([world_width / 2, world_width / 2])
circle_radius = world_width / 4

# ---------------------------------------------
# Cost Function Parameters
# ---------------------------------------------
cost_w1 = 0.006   # alignment error
cost_w2 = 6       # rise time squared
cost_w3 = 0.07    # control effort
psi_threshold = 0.6

# Global variables for cost accumulation
total_cost = 0.0
t_rise = num_steps
t_rise_recorded = False

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
    """
    Raycast from `position` in direction `heading + sensor_angle`.
    Returns the distance to the first obstacle intersection found,
    and a vector from the obstacle to the robot (for reference if needed).
    """
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
    """
    Adjust K and C based on how close the nearest obstacle is.
    """
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
    """
    Each robot uses 3 sensors: left (+30°), center (0°), right (-30°).
    - If left sensor sees obstacle => push to the right
    - If right sensor sees obstacle => push to the left
    - If center sensor sees obstacle => choose side with more space
    """
    num_robots = len(positions)
    forces = np.zeros((num_robots, 2))

    # sensor angles: left, center, right
    left_angle = np.radians(30)
    center_angle = 0.0
    right_angle = np.radians(-30)

    for i in range(num_robots):
        # Basic forces
        alignment_force = np.zeros(2)
        cohesion_force = target_positions[i] - positions[i]
        robot_repulsion_force = np.zeros(2)

        # Grab headings for left/right
        heading_dir = np.array([np.cos(headings[i]), np.sin(headings[i])])
        # left_dir: +90 deg rotation of heading_dir
        left_dir = np.array([-heading_dir[1], heading_dir[0]])
        # right_dir: -90 deg rotation of heading_dir
        right_dir = np.array([heading_dir[1], -heading_dir[0]])

        # --- Raycast sensors ---
        dist_left, _ = raycast_sensor(positions[i], headings[i], left_angle, obstacles, sensor_detection_distance)
        dist_center, _ = raycast_sensor(positions[i], headings[i], center_angle, obstacles, sensor_detection_distance)
        dist_right, _ = raycast_sensor(positions[i], headings[i], right_angle, obstacles, sensor_detection_distance)

        # min distance used to adapt K & C
        min_distance = min(dist_left, dist_center, dist_right)
        updated_K, updated_C = adjust_alignment_cohesion_gradual(current_K, current_C, min_distance)
        current_K = updated_K
        current_C = updated_C

        # -------------------------------
        # Build Obstacle Avoidance Force
        # -------------------------------
        # We can define a helper function for scaling:
        def avoidance_scale(d):
            # If d >= sensor_detection_distance => no push
            # otherwise scale up as it gets closer
            return beta * max(0.0, (sensor_detection_distance - d)) / sensor_detection_distance

        avoidance_force = np.zeros(2)

        # 1) If left sensor sees an obstacle => push right
        if dist_left < sensor_detection_distance:
            avoidance_force += avoidance_scale(dist_left) * right_dir

        # 2) If right sensor sees an obstacle => push left
        if dist_right < sensor_detection_distance:
            avoidance_force += avoidance_scale(dist_right) * left_dir

        # 3) If center sensor sees an obstacle => choose side with more space
        if dist_center < sensor_detection_distance:
            # If left side has more space => push left, else push right
            if dist_left > dist_right:
                # more room on left side => move left
                avoidance_force += avoidance_scale(dist_center) * left_dir
            else:
                # more room on right side => move right
                avoidance_force += avoidance_scale(dist_center) * right_dir

        # -------------------------------
        # Robot-Robot Repulsion
        # -------------------------------
        for j in range(num_robots):
            if i == j:
                continue
            direction = positions[i] - positions[j]
            distance = np.linalg.norm(direction)
            if distance < sensor_buffer_radius:
                effective_distance = max(distance, 1e-4)
                robot_repulsion_force += (direction / effective_distance) * (alpha / effective_distance)

        # -------------------------------
        # Alignment with neighbors
        # -------------------------------
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

        # Combine everything
        forces[i] = updated_C * cohesion_force \
                     + avoidance_force \
                     + updated_K * alignment_force \
                     + robot_repulsion_force

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
# Cost Function Computation
# -------------------------------------------
def compute_swarm_alignment(headings):
    avg_cos = np.mean(np.cos(headings))
    avg_sin = np.mean(np.sin(headings))
    psi = np.sqrt(avg_cos**2 + avg_sin**2)
    return psi


# -------------------------------------------
# Main Function
# -------------------------------------------
def main():
    # Initialize Swarm
    start_position = circle_center + circle_radius * np.array([1, 0])
    headings = np.random.uniform(0, 2 * np.pi, num_robots)
    velocities = np.zeros((num_robots, 2))

    if formation_type.lower() == "triangle":
        positions = initialize_positions_triangle(num_robots, start_position, formation_size_triangle_base)
        get_target_positions_fn = get_target_positions_triangle
    else:
        positions = initialize_positions(num_robots, start_position, formation_radius_base)
        get_target_positions_fn = get_target_positions

    # Obstacles
    obstacles = generate_varied_obstacles_with_levels(
        circle_center,
        circle_radius,
        num_obstacles,
        min_obstacle_size,
        max_obstacle_size,
        offset_degrees,
        passage_width,
        obstacle_level
    )

    global total_cost, t_rise, t_rise_recorded
    total_cost = 0.0
    t_rise = num_steps
    t_rise_recorded = False

    # Current alignment & cohesion
    current_K = K_base
    current_C = C_base

    fig, ax = plt.subplots()

    # Draw obstacles
    for obs in obstacles:
        if obs["type"] == "circle":
            circle_patch = plt.Circle(obs["position"], obs["radius"], color='red', fill=True)
            ax.add_artist(circle_patch)

    # Outline circle
    ax.add_artist(plt.Circle(circle_center, circle_radius, color='black', fill=False))

    scatter = ax.scatter(positions[:, 0], positions[:, 1], c='blue')
    ax.set_xlim(0, world_width)
    ax.set_ylim(0, world_width)
    ax.set_aspect('equal')
    count_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='darkred')

    cost_history = []

    formation_radius = formation_radius_base
    formation_size_triangle = formation_size_triangle_base
    alpha = alpha_base
    beta = beta_base

    moving_center_marker = ax.scatter([], [], color='green', marker='o', s=100, label="Moving Center")

    def animate(frame):
        global total_cost, t_rise, t_rise_recorded
        nonlocal positions, headings, velocities
        nonlocal formation_radius, formation_size_triangle, alpha, beta, current_K, current_C

        # 1) Check collisions
        updated_positions, removed_indices = check_collisions(positions, obstacles)
        if removed_indices:
            headings = np.delete(headings, removed_indices, axis=0)
            velocities = np.delete(velocities, removed_indices, axis=0)
            positions = updated_positions
        else:
            positions = updated_positions

        if len(positions) == 0:
            scatter.set_offsets([])
            count_text.set_text("Robots remaining: 0")
            if frame == num_steps - 1:
                plt.close(fig)
            return scatter,

        # 2) Adapt formation
        formation_radius, formation_size_triangle, alpha, beta = adapt_parameters(
            positions, obstacles,
            formation_radius_base, formation_size_triangle_base,
            alpha_base, beta_base
        )

        # 3) Get target positions
        moving_center = get_moving_center(frame, num_steps, positions)
        if formation_type.lower() == "triangle":
            target_positions = get_target_positions_triangle(moving_center, len(positions), formation_size_triangle)
        else:
            target_positions = get_target_positions(moving_center, len(positions), formation_radius)

        # 4) Compute forces (with new side-based sensor logic)
        forces, current_K, current_C = compute_forces_with_sensors(
            positions,
            headings,
            velocities,
            target_positions,
            obstacles,
            current_K,
            current_C,
            alpha,
            beta
        )

        # 5) Update positions & headings
        positions, headings = update_positions_and_headings(
            positions,
            headings,
            forces,
            robot_max_speed,
            (world_width, world_boundary_tolerance)
        )

        # 6) Update scatter and text
        scatter.set_offsets(positions)
        count_text.set_text(f"Robots remaining: {len(positions)}")

        # 7) Update moving center marker
        moving_center_marker.set_offsets([moving_center[0], moving_center[1]])

        # 8) Compute swarm alignment
        psi = compute_swarm_alignment(headings)

        # Record t_rise if threshold is met for the first time
        if (not t_rise_recorded) and (psi >= psi_threshold):
            t_rise = frame
            t_rise_recorded = True

        # 9) Compute sum of force magnitudes
        sum_force = np.sum(np.linalg.norm(forces, axis=1))

        # 10) Compute cost
        current_t_rise = t_rise if t_rise_recorded else num_steps
        cost_step = cost_w1 * (1 - psi)**2 + cost_w2 * (current_t_rise)**2 + cost_w3 * sum_force
        total_cost += cost_step
        cost_history.append(total_cost)

        # Record for plotting
        K_values_over_time.append(current_K)
        C_values_over_time.append(current_C)
        alpha_values_over_time.append(alpha)
        beta_values_over_time.append(beta)

        return scatter, moving_center_marker

    ani = animation.FuncAnimation(fig, animate, frames=num_steps, interval=100, repeat=True)
    plt.show()

    # End-of-run summary
    print(f"Robots left after simulation: {len(positions)}")
    print(f"Final accumulated cost: {total_cost:.2f}")
    print(f"Rise time (first frame where psi >= {psi_threshold}): {t_rise}")

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

if __name__ == "__main__":
    main()
