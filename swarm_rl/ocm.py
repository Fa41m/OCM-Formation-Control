import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------------------------
# Global Parameters
# ---------------------------------------------
formation_type = "triangle"  # "circle" or "triangle"

num_robots = 40
num_steps = 400

# Baseline repulsion strengths
alpha_base = 0.4    # attraction to the center
beta_base = 0.4     # repulsion from obstacles

# For circle formation
formation_radius_base = 3.0

# For triangle formation
formation_size_triangle_base = 1.0

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

# World & Speed
width = 60
max_speed = 0.3
world_boundary_tolerance = 0.5

# Obstacle Parameters
num_obstacles = 3
min_obstacle_size = 1.0
max_obstacle_size = 2.0
offset_degrees = 50
passage_width = 3.0
obstacle_level = 3

# Sensor Parameters
sensor_detection_distance = 4.0
sensor_buffer_radius = 1.0
object_sensor_buffer_radius = 0.5

# Circle center
circle_center = np.array([width / 2, width / 2])
circle_radius = width / 4

# Global collision threshold
collision_zone = 0.1

# ---------------------------------------------
# Helper Functions
# ---------------------------------------------
def generate_varied_obstacles_with_levels(
    center, radius, num_obstacles, min_size, max_size,
    offset_degrees, passage_width, level
):
    offset_radians = np.deg2rad(offset_degrees)
    angles = np.linspace(0, 2 * np.pi, num_obstacles, endpoint=False) + offset_radians
    obstacles = []

    for i, angle in enumerate(angles):
        if level == 1 or (level == 4 and i % 3 == 0):  # Offset obstacles
            offset_distance = np.random.choice([-1, 1]) * 4
            pos = center + (radius + offset_distance) * np.array([np.cos(angle), np.sin(angle)])
            size = np.random.uniform(1, 2.5)
            obstacles.append({"type": "circle", "position": pos, "radius": size})
        elif level == 2 or (level == 4 and i % 3 == 1):  # Obstacles on the circle
            pos = center + radius * np.array([np.cos(angle), np.sin(angle)])
            size = np.random.uniform(1.5, 2.5)
            obstacles.append({"type": "circle", "position": pos, "radius": size})
        elif level == 3 or (level == 4 and i % 3 == 2):  # Paired obstacles with a passage
            central_pos = center + radius * np.array([np.cos(angle), np.sin(angle)])
            size1 = np.random.uniform(2, 4)
            size2 = np.random.uniform(2, 4)
            adjusted_width = max(passage_width, size1 + size2 + passage_width)
            pos1 = central_pos - adjusted_width / 2 * np.array([np.cos(angle), np.sin(angle)])
            pos2 = central_pos + adjusted_width / 2 * np.array([np.cos(angle), np.sin(angle)])
            obstacles.extend([
                {"type": "circle", "position": pos1, "radius": size1},
                {"type": "circle", "position": pos2, "radius": size2}
            ])

    return obstacles


# ---------------
# Circle Formation
# ---------------
def initialize_positions(num_robots, start_position, formation_radius):
    angles = np.linspace(0, 2 * np.pi, num_robots, endpoint=False)
    return np.array([
        start_position + formation_radius * np.array([np.cos(angle), np.sin(angle)])
        for angle in angles
    ])

def get_moving_center(frame, total_frames):
    theta = 2 * np.pi * frame / total_frames
    return circle_center + circle_radius * np.array([np.cos(theta), np.sin(theta)])

def get_target_positions(moving_center, num_robots, formation_radius):
    angles = np.linspace(0, 2 * np.pi, num_robots, endpoint=False)
    return np.array([
        moving_center + formation_radius * np.array([np.cos(angle), np.sin(angle)])
        for angle in angles
    ])

# ---------------
# Triangle Formation
# ---------------
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

# ---------------
# Sensor / Avoidance
# ---------------
def raycast_sensor(position, heading, sensor_angle, obstacles, sensor_detection_distance):
    sensor_direction = np.array([
        np.cos(heading + sensor_angle),
        np.sin(heading + sensor_angle)
    ])
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
                    np.sqrt((obs_radius + object_sensor_buffer_radius)**2
                            - distance_to_obstacle**2)
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
    if abs(new_value - target_value) < 0.02:
        new_value = target_value
    return new_value

def adjust_alignment_cohesion_gradual(current_K, current_C, center_dist):
    if center_dist < sensor_detection_distance:
        target_K = max(K_min, K_base * (center_dist / sensor_detection_distance))
        target_C = max(C_min, C_base * (center_dist / sensor_detection_distance))
    else:
        target_K = K_base
        target_C = C_base

    new_K = smooth_transition_with_bounds(current_K, target_K, lambda_K, K_min, K_max)
    new_C = smooth_transition_with_bounds(current_C, target_C, lambda_C, C_min, C_max)
    return new_K, new_C

# ---------------------------------------------
# Adaptation of formation, alpha, beta
# ---------------------------------------------
def adapt_parameters(
    positions, obstacles,
    base_formation_radius, base_formation_size_triangle,
    base_alpha, base_beta,
    min_dist_threshold=4.0
):
    if len(positions) == 0:
        return (base_formation_radius,
                base_formation_size_triangle,
                base_alpha, base_beta)

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

    # Find robot density
    min_robot_dist = float('inf')
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_robot_dist:
                min_robot_dist = d

    # If an obstacle is close, form shrink
    obs_factor = 0.0
    if min_obstacle_dist < min_dist_threshold:
        obs_factor = (min_dist_threshold - min_obstacle_dist) / min_dist_threshold
        obs_factor = np.clip(obs_factor, 0.0, 1.0)

    # Possibly shrink more if the obstacle is large
    large_obstacle_threshold = 3.0
    if obstacle_radius_for_closest > large_obstacle_threshold:
        obs_factor = min(obs_factor * 1.2, 1.0)

    # Shrink up to 50%
    shrink_ratio = 1.0 - 0.5 * obs_factor
    new_formation_radius = base_formation_radius * shrink_ratio
    new_formation_size_triangle = base_formation_size_triangle * shrink_ratio

    # Increase alpha if too dense
    density_factor = 0.0
    density_threshold = 1.0
    if min_robot_dist < density_threshold:
        density_factor = (density_threshold - min_robot_dist) / density_threshold
        density_factor = np.clip(density_factor, 0.0, 1.0)

    new_alpha = base_alpha * (1.0 + 0.5 * density_factor)
    new_beta = base_beta * (1.0 + 0.5 * obs_factor)

    return (
        new_formation_radius,
        new_formation_size_triangle,
        new_alpha,
        new_beta
    )

# ---------------------------------------------
# Force Computation
# ---------------------------------------------
def compute_forces_with_sensors(
    positions, headings, velocities, target_positions,
    obstacles, current_K, current_C, alpha, beta
):
    num_robots = len(positions)
    forces = np.zeros((num_robots, 2))

    for i in range(num_robots):
        alignment_force = np.zeros(2)
        cohesion_force = target_positions[i] - positions[i]
        avoidance_force = np.zeros(2)
        robot_repulsion_force = np.zeros(2)

        # Sensor
        center_distance, center_repulsion = raycast_sensor(
            positions[i], headings[i], 0, obstacles, sensor_detection_distance
        )

        # Adjust K/C
        updated_K, updated_C = adjust_alignment_cohesion_gradual(current_K, current_C, center_distance)
        current_K = updated_K
        current_C = updated_C

        # Obstacle avoidance
        if center_distance < sensor_detection_distance:
            effective_distance = max(center_distance - sensor_buffer_radius, 1e-6)
            avoidance_force += center_repulsion * (beta / effective_distance)

        # Robot-robot repulsion
        for j in range(num_robots):
            if i == j:
                continue
            direction = positions[i] - positions[j]
            distance = np.linalg.norm(direction)
            if distance < sensor_buffer_radius:
                effective_distance = max(distance, 1e-6)
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

        forces[i] = (
            current_C * cohesion_force +
            beta * avoidance_force +
            current_K * alignment_force +
            robot_repulsion_force
        )

    return forces, current_K, current_C

# ---------------------------------------------
# Collision Checking
# ---------------------------------------------
def check_collisions(positions, obstacles):
    """
    Return the updated positions (with collided robots removed),
    plus the list of indices that were removed.
    """
    to_remove = set()
    # Check robot-robot collisions
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < collision_zone:
                to_remove.add(i)
                to_remove.add(j)

    # Check robot-obstacle collisions
    for i, pos in enumerate(positions):
        for obs in obstacles:
            obs_pos = obs["position"]
            obs_radius = obs["radius"]
            if np.linalg.norm(pos - obs_pos) < (obs_radius + collision_zone):
                to_remove.add(i)

    to_remove_list = sorted(to_remove)  # sorted list of removed indices
    updated_positions = np.delete(positions, to_remove_list, axis=0)
    return updated_positions, to_remove_list

def enforce_boundary_conditions(positions, width, world_boundary_tolerance):
    return np.clip(positions, world_boundary_tolerance, width - world_boundary_tolerance)


def update_positions_and_headings(positions, headings, forces, max_speed, boundary_conditions):
    for i in range(len(positions)):
        velocity = forces[i]
        speed = np.linalg.norm(velocity)
        if speed > max_speed:
            velocity = (max_speed / speed) * velocity

        positions[i] += velocity

        if np.linalg.norm(velocity) > 1e-6:
            headings[i] = np.arctan2(velocity[1], velocity[0])

    positions = enforce_boundary_conditions(positions, *boundary_conditions)
    return positions, headings

# -------------------------------------------
# Main Function
# -------------------------------------------
def main():
    global alpha_base, beta_base
    global formation_radius_base, formation_size_triangle_base

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
        circle_center, circle_radius,
        num_obstacles, min_obstacle_size, max_obstacle_size,
        offset_degrees, passage_width, obstacle_level
    )

    # Current alignment & cohesion
    current_K = K_base
    current_C = C_base

    fig, ax = plt.subplots()

    # Draw obstacles
    for obs in obstacles:
        if obs["type"] == "circle":
            circle_patch = plt.Circle(obs["position"], obs["radius"], color='red', fill=True)
            ax.add_artist(circle_patch)

    ax.add_artist(plt.Circle(circle_center, circle_radius, color='black', fill=False))

    scatter = ax.scatter(positions[:, 0], positions[:, 1], c='blue')
    ax.set_xlim(0, width)
    ax.set_ylim(0, width)
    ax.set_aspect('equal')

    count_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='darkred')

    radial_errors = []
    formation_radius = formation_radius_base
    formation_size_triangle = formation_size_triangle_base
    alpha = alpha_base
    beta = beta_base

    def animate(frame):
        nonlocal positions, headings, velocities, current_K, current_C
        nonlocal formation_radius, formation_size_triangle, alpha, beta

        # 1) Check collisions, remove collided robots
        updated_positions, removed_indices = check_collisions(positions, obstacles)

        if removed_indices:
            print(f"Frame {frame}: removing {len(removed_indices)} collided robots.")
            # Remove corresponding rows from headings, velocities
            updated_headings = np.delete(headings, removed_indices, axis=0)
            updated_velocities = np.delete(velocities, removed_indices, axis=0)

            # Update them
            positions = updated_positions
            headings = updated_headings
            velocities = updated_velocities
        else:
            positions = updated_positions  # might be unchanged

        if len(positions) == 0:
            # All robots gone -> just set empty scatter, but continue frames
            scatter.set_offsets([])
            count_text.set_text("Robots remaining: 0")
            # Exit early by closing the figure if all robots are gone
            if frame == num_steps - 1:
                plt.close(fig)
            return scatter,

        # 2) Adapt formation
        formation_radius, formation_size_triangle, alpha, beta = adapt_parameters(
            positions, obstacles,
            formation_radius_base, formation_size_triangle_base,
            alpha_base, beta_base,
            min_dist_threshold=4.0
        )

        # 3) Get target positions
        moving_center = get_moving_center(frame, num_steps)
        if formation_type.lower() == "triangle":
            target_positions = get_target_positions_triangle(
                moving_center, len(positions), formation_size_triangle
            )
        else:
            target_positions = get_target_positions(
                moving_center, len(positions), formation_radius
            )

        # 4) Compute forces
        forces, current_K, current_C = compute_forces_with_sensors(
            positions, headings, velocities,
            target_positions, obstacles,
            current_K, current_C,
            alpha, beta
        )

        # 5) Update positions & headings
        positions, headings = update_positions_and_headings(
            positions, headings, forces, max_speed,
            (width, world_boundary_tolerance)
        )

        # 6) Update scatter
        scatter.set_offsets(positions)
        count_text.set_text(f"Robots remaining: {len(positions)}")

        # 7) Track radial error
        distances_from_center = np.linalg.norm(positions - circle_center, axis=1)
        radial_error = np.mean(np.abs(distances_from_center - circle_radius))
        radial_errors.append(radial_error)

        # 8) Track K/C
        K_values_over_time.append(current_K)
        C_values_over_time.append(current_C)
        alpha_values_over_time.append(alpha)
        beta_values_over_time.append(beta)

        return scatter,

    ani = animation.FuncAnimation(
        fig, animate, frames=num_steps, interval=100, repeat=True
    )
    plt.show()

    # End-of-run summary
    print(f"Robots left after simulation: {len(positions)}")

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

    # Plot radial errors
    plt.figure(figsize=(10, 6))
    plt.plot(radial_errors, label='Radial Error', color='red')
    plt.xlabel('Time Step')
    plt.ylabel('Radial Error (Distance from Circle Center)')
    plt.title('Radial Error Over Time')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
