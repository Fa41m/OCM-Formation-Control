import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Global Parameters
num_robots = 20
num_steps = 400
# Attraction strength
alpha = 0.4
# Repulsion strength
beta = 0.4
# Alignment strength
K = 0.2
# Cohesion strength
C = 0.1
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

def detect_with_sensor(position, heading, sensor_angle, obstacles, robots, sensor_detection_distance):
    """
    Detect obstacles or robots using a sensor at a given angle.
    """
    sensor_direction = np.array([
        np.cos(heading + sensor_angle),
        np.sin(heading + sensor_angle)
    ])
    sensor_position = position + sensor_direction * sensor_detection_distance

    # Check for obstacles
    for obs in obstacles:
        obs_pos = obs["position"]
        obs_radius = obs["radius"]
        distance = np.linalg.norm(sensor_position - obs_pos)
        if distance <= obs_radius + sensor_buffer_radius:
            return True

    # Check for other robots
    for robot_pos in robots:
        if np.array_equal(robot_pos, position):  # Skip self
            continue
        distance = np.linalg.norm(sensor_position - robot_pos)
        if distance <= sensor_buffer_radius:
            return True

    return False

def compute_forces_with_sensors(positions, headings, velocities, target_positions, obstacles):
    """
    Compute forces based on sensor readings, alignment, and cohesion.
    """
    num_robots = len(positions)
    forces = np.zeros((num_robots, 2))
    desired_velocities = np.zeros((num_robots, 2))

    for i in range(num_robots):
        # Initialize forces
        alignment_force = np.zeros(2)
        cohesion_force = target_positions[i] - positions[i]
        avoidance_force = np.zeros(2)
        repulsion_force = np.zeros(2)

        # Sensor detection
        left_sensor_angle = np.deg2rad(30)  # Left offset angle
        right_sensor_angle = -np.deg2rad(30)  # Right offset angle

        # Check sensors for obstacles
        left_detected = detect_with_sensor(positions[i], headings[i], left_sensor_angle, obstacles, positions, sensor_detection_distance)
        center_detected = detect_with_sensor(positions[i], headings[i], 0, obstacles, positions, sensor_detection_distance)
        right_detected = detect_with_sensor(positions[i], headings[i], right_sensor_angle, obstacles, positions, sensor_detection_distance)

        # Adjust avoidance force based on sensor readings
        if left_detected and not right_detected:
            avoidance_force += np.array([np.cos(headings[i] - right_sensor_angle), np.sin(headings[i] - right_sensor_angle)])
        elif right_detected and not left_detected:
            avoidance_force += np.array([np.cos(headings[i] + left_sensor_angle), np.sin(headings[i] + left_sensor_angle)])
        elif center_detected:
            avoidance_force += -np.array([np.cos(headings[i]), np.sin(headings[i])])

        # Obstacle repulsion force
        for obs in obstacles:
            obs_pos = obs["position"]
            obs_radius = obs["radius"]
            distance = np.linalg.norm(positions[i] - obs_pos)
            if distance < obs_radius + sensor_buffer_radius:
                direction_away = (positions[i] - obs_pos) / (distance + 1e-6)
                magnitude = beta / (distance - obs_radius + 1e-6)
                repulsion_force += magnitude * direction_away

        # Robot repulsion force
        for j in range(num_robots):
            if i == j:
                continue
            distance = np.linalg.norm(positions[i] - positions[j])
            if distance < sensor_buffer_radius:
                direction_away = (positions[i] - positions[j]) / (distance + 1e-6)
                magnitude = alpha / (distance + 1e-6)
                repulsion_force += magnitude * direction_away

        # Alignment force (match orientations with neighbors)
        neighbors = [
            j for j in range(num_robots)
            if i != j and np.linalg.norm(positions[i] - positions[j]) < sensor_detection_distance
        ]
        if neighbors:
            avg_heading = np.mean([headings[j] for j in neighbors])
            alignment_force = np.array([
                np.cos(avg_heading) - np.cos(headings[i]),
                np.sin(avg_heading) - np.sin(headings[i])
            ])

        # Velocity matching
        if neighbors:
            avg_velocity = np.mean([velocities[j] for j in neighbors], axis=0)
            desired_velocities[i] = avg_velocity

        # Combine forces
        forces[i] = (
            alpha * cohesion_force +
            beta * avoidance_force +
            K * alignment_force +
            repulsion_force
        )

    return forces, desired_velocities


# To make sure the robots do not go out of the boundary of the world
def enforce_boundary_conditions(positions, width, world_boundary_tolerance):
    positions = np.clip(positions, world_boundary_tolerance, width - world_boundary_tolerance)
    return positions

def update_positions_and_headings(positions, headings, velocities, forces, max_speed, boundary_conditions):
    for i in range(len(positions)):
        velocity = forces[i]
        speed = np.linalg.norm(velocity)
        if speed > max_speed:
            velocity = max_speed * velocity / speed
        velocities[i] = velocity
        positions[i] += velocities[i]
        headings[i] = np.arctan2(velocity[1], velocity[0])
    positions = enforce_boundary_conditions(positions, *boundary_conditions)
    return positions, headings, velocities

def animate(frame, positions, headings, velocities, formation_radius, obstacles, scatter):
    moving_center = get_moving_center(frame, num_steps)
    target_positions = get_target_positions(moving_center, num_robots, formation_radius)
    forces, desired_velocities = compute_forces_with_sensors(positions, headings, velocities, target_positions, obstacles)
    positions[:], headings[:], velocities[:] = update_positions_and_headings(positions, headings, velocities, forces, max_speed, (width, world_boundary_tolerance))
    scatter.set_offsets(positions)
    return scatter,

# Main Function
def main():
    # Initialize Swarm and Obstacles
    formation_radius = max(sensor_buffer_radius * num_robots / (2 * np.pi), sensor_buffer_radius)
    start_position = circle_center + circle_radius * np.array([1, 0])
    positions = initialize_positions(num_robots, start_position, formation_radius)
    headings = np.random.uniform(0, 2 * np.pi, num_robots)
    velocities = np.zeros_like(positions)
    obstacle_level = 3
    obstacles = generate_varied_obstacles_with_levels(
        circle_center, circle_radius, num_obstacles, min_obstacle_size, max_obstacle_size,
        offset_degrees, passage_width, obstacle_level
    )
    # Set up the plot
    fig, ax = plt.subplots()
    # plot obstacles
    for obs in obstacles:
        if obs["type"] == "circle":
            circle = plt.Circle(obs["position"], obs["radius"], color='red', fill=True)
            ax.add_artist(circle)
    # plot circle path
    ax.add_artist(plt.Circle(circle_center, circle_radius, color='black', fill=False))
    scatter = ax.scatter(positions[:, 0], positions[:, 1], c='blue', label='Robots')
    ax.set_xlim(0, width)
    ax.set_ylim(0, width)
    ax.set_aspect('equal')

    # Animation
    ani = animation.FuncAnimation(
        fig, animate, frames=num_steps, interval=100, repeat=True,
        fargs=(positions, headings, velocities, formation_radius, obstacles, scatter)
    )
    plt.show()

if __name__ == "__main__":
    main()
