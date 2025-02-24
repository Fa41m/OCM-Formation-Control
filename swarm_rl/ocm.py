import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Global Parameters

formation_type = "circle"  # <<--- CHOOSE "circle" or "triangle" here

num_robots = 20
num_steps = 400
# Attraction strength
alpha = 0.4
# Repulsion strength
beta = 0.4

obstacle_level = 3

K_base = 0.7  # Base alignment strength
C_base = 0.6  # Base cohesion strength
K_min, K_max = 0.005, 0.995  # Range for alignment strength
C_min, C_max = 0.005, 0.995  # Range for cohesion strength

# Arrays to store K and C values over time
K_values_over_time = []
C_values_over_time = []

# Smoothing factor for gradual increase and decrease
lambda_K = 0.7  # Controls the smoothness for alignment strength
lambda_C = 0.5  # Controls the smoothness for cohesion strength

# Width of the 2D space (world boundary)
width = 60
# Speed of the robots
constant_speed = 0.1
max_speed = 0.3
world_boundary_tolerance = 0.5

# Obstacle Parameters
num_obstacles = 3
min_obstacle_size = 1.0
max_obstacle_size = 2.0
offset_degrees = 50
passage_width = 3.0

# Sensor Parameters
sensor_angle_degrees = 30.0
sensor_angle_radians = np.deg2rad(sensor_angle_degrees)
# This variable is used to determine the distance at which the sensor can detect obstacles or other robots
sensor_detection_distance = 4.0
# There should be a minimum of 1 unit buffer between the sensor and the obstacle at all times
sensor_buffer_radius = 1.0
object_sensor_buffer_radius = 0.5

# Center and Radius for Circular Path
circle_center = np.array([width / 2, width / 2])
circle_radius = width / 4

# Global Variables for Logging
K_values = []
C_values = []

# Global collision zone threshold
collision_zone = 0.1  # Threshold for collision detection

# For circle formation
formation_radius = max(sensor_buffer_radius * num_robots / (2 * np.pi), sensor_buffer_radius)
# For triangle formation (you can adjust as needed)
formation_size_triangle = 1.0  # spacing scale for the triangle

# Helper Functions
def generate_varied_obstacles_with_levels(center, radius, num_obstacles, min_size, max_size, offset_degrees, passage_width, level):
    """
    Generate obstacles based on the specified level.
    Level 0: No obstacles.
    Level 1: Offset obstacles.
    Level 2: Obstacles on the circle.
    Level 3: Paired obstacles with a passage.
    Level 4: All combined.
    """
    offset_radians = np.deg2rad(offset_degrees)
    angles = np.linspace(0, 2 * np.pi, num_obstacles, endpoint=False) + offset_radians
    obstacles = []

    for i, angle in enumerate(angles):
        if level == 1 or (level == 4 and i % 3 == 0):  # Offset obstacles
            offset_distance = np.random.choice([-1, 1]) * 4
            pos = center + (radius + offset_distance) * np.array([np.cos(angle), np.sin(angle)])
            # size = np.random.uniform(min_size, max_size)
            size = np.random.uniform(1, 2.5)
            obstacles.append({"type": "circle", "position": pos, "radius": size})
        elif level == 2 or (level == 4 and i % 3 == 1):  # Obstacles on the circle
            pos = center + radius * np.array([np.cos(angle), np.sin(angle)])
            # size = np.random.uniform(min_size, max_size)
            size = np.random.uniform(1.5, 2.5)
            obstacles.append({"type": "circle", "position": pos, "radius": size})
        elif level == 3 or (level == 4 and i % 3 == 2):  # Paired obstacles with a passage
            central_pos = center + radius * np.array([np.cos(angle), np.sin(angle)])
            # size1 = np.random.uniform(min_size, max_size)
            # size2 = np.random.uniform(min_size, max_size)
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

# Circle Formation
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
    
# Triangle Formation
def initialize_positions_triangle(num_robots, start_position, formation_size):
    # Compute heading angle normal to the circle at start_position
    radius_vector = start_position - circle_center
    heading_vector = radius_vector  # Normal vector
    heading_angle = np.arctan2(heading_vector[1], heading_vector[0])

    # Determine rows needed for the triangle
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
    h_spacing = formation_size * 1.5  # vertical spacing between rows
    v_spacing = formation_size       # horizontal spacing
    y_offset = 0

    for i, bots_in_row in enumerate(rows):
        x_offset = -v_spacing * (bots_in_row - 1) / 2.0
        for j in range(bots_in_row):
            positions_relative.append([x_offset + j * v_spacing, y_offset])
        y_offset -= h_spacing

    positions_relative = np.array(positions_relative)

    # Rotate by heading_angle
    rotation_matrix = np.array([
        [np.cos(heading_angle), -np.sin(heading_angle)],
        [np.sin(heading_angle),  np.cos(heading_angle)]
    ])
    rotated_positions = positions_relative @ rotation_matrix.T

    # Translate to start_position
    initial_positions = rotated_positions + start_position
    return initial_positions

def get_target_positions_triangle(moving_center, num_robots, formation_size):
    radius_vector = moving_center - circle_center
    heading_vector = radius_vector
    heading_angle = np.arctan2(heading_vector[1], heading_vector[0])

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

    for i, bots_in_row in enumerate(rows):
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

    target_positions = rotated_positions + moving_center
    return target_positions

# TODO: Combine the sensors to get a better result for the robots to avoid obstacles
def raycast_sensor(position, heading, sensor_angle, obstacles, sensor_detection_distance):
    """
    Simulate a sensor raycast to detect obstacles along a given direction.
    Returns the distance to the nearest obstacle or infinity if none are detected.
    """
    sensor_direction = np.array([
        np.cos(heading + sensor_angle),
        np.sin(heading + sensor_angle)
    ])
    sensor_start = position
    sensor_end = sensor_start + sensor_direction * sensor_detection_distance

    min_distance = sensor_detection_distance  # Default to max sensor range
    repulsion_vector = np.zeros(2)  # Vector to apply repulsion force

    for obs in obstacles:
        obs_pos = obs["position"]
        obs_radius = obs["radius"]

        # Vector from sensor start to obstacle center
        to_obstacle = obs_pos - sensor_start

        # Projection of the obstacle vector onto the sensor direction
        projection_length = np.dot(to_obstacle, sensor_direction)

        if 0 < projection_length < sensor_detection_distance:
            # Closest point on the sensor ray to the obstacle center
            closest_point = sensor_start + projection_length * sensor_direction
            distance_to_obstacle = np.linalg.norm(closest_point - obs_pos)

            # Check if the ray intersects the obstacle + buffer radius
            if distance_to_obstacle <= obs_radius + object_sensor_buffer_radius:
                intersection_distance = projection_length - np.sqrt(
                    (obs_radius + object_sensor_buffer_radius)**2 - distance_to_obstacle**2
                )
                if intersection_distance < min_distance:
                    min_distance = intersection_distance
                    repulsion_vector = (sensor_start - obs_pos) / (np.linalg.norm(sensor_start - obs_pos) + 1e-6)

    return min_distance, repulsion_vector

def detect_with_sensor(position, heading, sensor_angle, robots, sensor_detection_distance):
    """
    Detect obstacles using a sensor at a given angle relative to the robot's heading.
    """
    sensor_direction = np.array([
        np.cos(heading + sensor_angle),
        np.sin(heading + sensor_angle)
    ])
    sensor_position = position + sensor_direction * sensor_detection_distance

    # Check for other robots (or obstacles, if simulated as robots)
    for robot_pos in robots:
        if np.array_equal(robot_pos, position):  # Skip self
            continue
        distance = np.linalg.norm(sensor_position - robot_pos)
        if distance <= sensor_buffer_radius:
            return True

    return False

# Gradual transition function with bounds and early stopping
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

# TODO: Solution to the robots crashing into each other but sometimes it isn't fluid
def compute_forces_with_sensors(positions, headings, velocities, target_positions, obstacles, current_K, current_C):
    num_robots = len(positions)
    forces = np.zeros((num_robots, 2))
    desired_velocities = np.zeros((num_robots, 2))
    updated_K_values = np.full(num_robots, current_K)
    updated_C_values = np.full(num_robots, current_C)
    collisions = set()  # Store indices of robots involved in collisions
    collision_threshold = collision_zone  # Threshold for detecting collisions

    for i in range(num_robots):
        alignment_force = np.zeros(2)
        cohesion_force = target_positions[i] - positions[i]
        avoidance_force = np.zeros(2)
        robot_repulsion_force = np.zeros(2)

        # Obstacle detection via sensors
        center_distance, center_repulsion = raycast_sensor(positions[i], headings[i], 0, obstacles, sensor_detection_distance)

        # Gradual adjustment of K and C
        updated_K_values[i], updated_C_values[i] = adjust_alignment_cohesion_gradual(updated_K_values[i], updated_C_values[i], center_distance)

        # Avoidance force for obstacles
        if center_distance < sensor_detection_distance:
            effective_distance = max(center_distance - sensor_buffer_radius, 1e-6)
            avoidance_force += center_repulsion * (beta / effective_distance)

        # Repulsion force and safety boundary between robots
        for j in range(num_robots):
            if i == j:
                continue
            direction = positions[i] - positions[j]
            distance = np.linalg.norm(direction)

            if distance < sensor_buffer_radius:
                effective_distance = max(distance, 1e-6)
                robot_repulsion_force += (direction / effective_distance) * (alpha / effective_distance)

                # Detect collisions between robots
                if distance < collision_threshold:  # Define collision_threshold for collisions
                    collisions.update([i, j])  # Add both robots to the collision set

        # Alignment force (average heading of neighbors)
        neighbors = [j for j in range(num_robots) if i != j and np.linalg.norm(positions[i] - positions[j]) < sensor_detection_distance]
        if neighbors:
            avg_heading = np.mean([headings[j] for j in neighbors])
            alignment_force = np.array([np.cos(avg_heading) - np.cos(headings[i]), np.sin(avg_heading) - np.sin(headings[i])])

        # Combine forces
        forces[i] = (
            updated_C_values[i] * cohesion_force +
            beta * avoidance_force +
            updated_K_values[i] * alignment_force +
            robot_repulsion_force
        )

    return forces, desired_velocities, np.mean(updated_K_values), np.mean(updated_C_values), collisions

# To make sure the robots do not go out of the boundary of the world
def enforce_boundary_conditions(positions, width, world_boundary_tolerance):
    positions = np.clip(positions, world_boundary_tolerance, width - world_boundary_tolerance)
    return positions

def update_positions_and_headings(positions, headings, forces, max_speed, boundary_conditions):
    for i in range(len(positions)):
        # Limit speed
        velocity = forces[i]
        speed = np.linalg.norm(velocity)
        if speed > max_speed:
            velocity = max_speed * velocity / speed

        # Update position
        positions[i] += velocity

        # Update heading based on velocity direction
        if np.linalg.norm(velocity) > 1e-6:  # Avoid divide-by-zero
            desired_heading = np.arctan2(velocity[1], velocity[0])
            headings[i] = desired_heading

    # Enforce boundary conditions
    positions = enforce_boundary_conditions(positions, *boundary_conditions)
    return positions, headings


def check_collisions(positions, obstacles):
    """
    Checks if any robot collides with another robot or an obstacle.
    Removes the robot from the simulation if a collision is detected.
    """
    to_remove = set()

    # Check for collisions between robots
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            distance = np.linalg.norm(positions[i] - positions[j])
            if distance < collision_zone:
                to_remove.add(i)
                to_remove.add(j)

    # Check for collisions with obstacles
    for i, pos in enumerate(positions):
        for obs in obstacles:
            obs_pos = obs["position"]
            obs_radius = obs["radius"]
            if np.linalg.norm(pos - obs_pos) < (obs_radius + collision_zone):
                to_remove.add(i)

    # Remove colliding robots
    positions = np.delete(positions, list(to_remove), axis=0)
    return positions, len(to_remove)

def animate(frame, positions, headings, velocities, formation_radius, obstacles, scatter):
    moving_center = get_moving_center(frame, num_steps)
    target_positions = get_target_positions(moving_center, num_robots, formation_radius)
    forces, desired_velocities = compute_forces_with_sensors(positions, headings, velocities, target_positions, obstacles)
    positions[:], headings[:] = update_positions_and_headings(positions, headings, forces, max_speed, (width, world_boundary_tolerance))
    scatter.set_offsets(positions)
    return scatter,

# Main Function
def main():
    # Initialize Swarm and Obstacles
    start_position = circle_center + circle_radius * np.array([1, 0])
    # positions = initialize_positions(num_robots, start_position, formation_radius)
    headings = np.random.uniform(0, 2 * np.pi, num_robots)
    # Depending on formation_type, choose how to initialize
    if formation_type.lower() == "triangle":
        # Triangle formation
        positions = initialize_positions_triangle(num_robots, start_position, formation_size_triangle)
        get_target_positions_fn = get_target_positions_triangle
    else:
        # Circle formation (default)
        positions = initialize_positions(num_robots, start_position, formation_radius)
        get_target_positions_fn = get_target_positions
    velocities = np.zeros_like(positions)
    obstacles = generate_varied_obstacles_with_levels(
        circle_center, circle_radius, num_obstacles, min_obstacle_size, max_obstacle_size,
        offset_degrees, passage_width, obstacle_level
    )
    current_K = K_base
    current_C = C_base

    # Set up the plot
    fig, ax = plt.subplots()
    for obs in obstacles:
        if obs["type"] == "circle":
            circle = plt.Circle(obs["position"], obs["radius"], color='red', fill=True)
            ax.add_artist(circle)
    ax.add_artist(plt.Circle(circle_center, circle_radius, color='black', fill=False))
    scatter = ax.scatter(positions[:, 0], positions[:, 1], c='blue', label='Robots')
    ax.set_xlim(0, width)
    ax.set_ylim(0, width)
    ax.set_aspect('equal')

    # Add text to show the number of robots remaining
    count_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='darkred')
    
    radial_errors = []  # List to store radial error at each timestep

    # Animation function
    def animate(frame):
        nonlocal positions, headings, current_K, current_C

        moving_center = get_moving_center(frame, num_steps)
        target_positions = get_target_positions_fn(moving_center, len(positions), formation_radius if formation_type=="circle" else formation_size_triangle)

        # Check for collisions and update positions
        positions[:], num_removed = check_collisions(positions, obstacles)
        if num_removed > 0:
            print(f"{num_removed} robots removed due to collisions at frame {frame}.")

        if len(positions) == 0:
            print("All robots have crashed.")
            plt.close(fig)
            return scatter,

        forces, _, current_K, current_C, collisions = compute_forces_with_sensors(
            positions, headings, velocities, target_positions, obstacles, current_K, current_C
        )
        positions[:], headings[:] = update_positions_and_headings(positions, headings, forces, max_speed, (width, world_boundary_tolerance))
        scatter.set_offsets(positions)

        # Update the count text
        count_text.set_text(f"Robots remaining: {len(positions)}")
        # count_text.set_text(f"Robots remaining: {num_robots - len(collisions)}")
        
        # Calculate radial error
        distances_from_center = np.linalg.norm(positions - circle_center, axis=1)
        radial_error = np.mean(np.abs(distances_from_center - circle_radius))
        radial_errors.append(radial_error)  # Store the radial error for this timestep

        K_values_over_time.append(current_K)
        C_values_over_time.append(current_C)

        return scatter,

    ani = animation.FuncAnimation(
        fig, animate, frames=num_steps, interval=100, repeat=True
    )
    plt.show()

    # Output remaining robots after the simulation ends
    print(f"Number of robots remaining after the simulation: {len(positions)}")

    # Plot K and C values over time
    plt.figure(figsize=(10, 6))
    plt.plot(K_values_over_time, label='Alignment Strength (K)', color='blue')
    plt.plot(C_values_over_time, label='Cohesion Strength (C)', color='green')
    plt.xlabel('Time Step')
    plt.ylabel('Strength Values')
    plt.title('Alignment (K) and Cohesion (C) Strengths Over Time')
    plt.legend()
    plt.show()
    
    # Plot Radial Error Over Time
    plt.figure(figsize=(10, 6))
    plt.plot(radial_errors, label='Radial Error', color='red')
    plt.xlabel('Time Step')
    plt.ylabel('Radial Error (Distance from Circle)')
    plt.title('Radial Error Over Time')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
