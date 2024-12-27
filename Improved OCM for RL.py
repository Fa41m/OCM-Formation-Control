import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Global Parameters
num_robots = 5
num_steps = 400
alpha = 0.2
beta = 0.2
K = 0.2
C = 0.1
width = 45
buffer_radius = 1
sensing_radius = 1
constant_speed = 0.1
max_speed = 0.3
num_checkpoints = 10
boundary_tolerance = 0.5

# Obstacle Parameters
num_obstacles = 3
min_obstacle_size = 1.0
max_obstacle_size = 2.0
offset_degrees = 50
passage_width = 2.0

sensor_angle_degrees = 30.0
sensor_angle_radians = np.deg2rad(sensor_angle_degrees)
detection_distance = 0.7

# Center and Radius for Circular Path
circle_center = np.array([width / 2, width / 2])
circle_radius = width / 4

# Global Variables for Logging
K_values = []
C_values = []

# Helper Functions
def generate_varied_obstacles_with_levels(center, radius, num_obstacles, min_size, max_size, offset_degrees, passage_width):
    offset_radians = np.deg2rad(offset_degrees)
    angles = np.linspace(0, 2 * np.pi, num_obstacles, endpoint=False) + offset_radians
    obstacles = []
    for i, angle in enumerate(angles):
        if i % 3 == 0:
            offset_distance = np.random.choice([-1, 1]) * 2
            pos = center + (radius + offset_distance) * np.array([np.cos(angle), np.sin(angle)])
            size = np.random.uniform(min_size, max_size)
            obstacles.append({"type": "circle", "position": pos, "radius": size})
        elif i % 3 == 1:
            pos = center + radius * np.array([np.cos(angle), np.sin(angle)])
            size = np.random.uniform(min_size, max_size)
            obstacles.append({"type": "circle", "position": pos, "radius": size})
        elif i % 3 == 2:
            central_pos = center + radius * np.array([np.cos(angle), np.sin(angle)])
            size1 = np.random.uniform(min_size, max_size)
            size2 = np.random.uniform(min_size, max_size)
            adjusted_width = max(passage_width, size1 + size2 + 1.5)
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

def compute_forces(positions, headings, target_positions, obstacles):
    num_robots = len(positions)
    forces = np.zeros((num_robots, 2))
    for i in range(num_robots):
        cohesion_force = target_positions[i] - positions[i]
        obstacle_avoidance_force = np.zeros(2)
        for obs in obstacles:
            obs_pos = obs["position"]
            obs_radius = obs["radius"]
            distance = np.linalg.norm(positions[i] - obs_pos)
            if distance < obs_radius + buffer_radius:
                obstacle_avoidance_force += (positions[i] - obs_pos) / (distance**2 + 1e-6)
        forces[i] = alpha * cohesion_force + beta * obstacle_avoidance_force
    return forces

def update_positions(positions, forces, max_speed, boundary_conditions):
    for i in range(len(positions)):
        velocity = forces[i]
        speed = np.linalg.norm(velocity)
        if speed > max_speed:
            velocity = max_speed * velocity / speed
        positions[i] += velocity
    positions = enforce_boundary_conditions(positions, *boundary_conditions)
    return positions

def enforce_boundary_conditions(positions, width, boundary_tolerance):
    positions = np.clip(positions, boundary_tolerance, width - boundary_tolerance)
    return positions

def animate(frame, positions, headings, formation_radius, obstacles, scatter):
    moving_center = get_moving_center(frame, num_steps)
    target_positions = get_target_positions(moving_center, num_robots, formation_radius)
    forces = compute_forces(positions, headings, target_positions, obstacles)
    positions[:] = update_positions(positions, forces, max_speed, (width, boundary_tolerance))
    scatter.set_offsets(positions)
    return scatter,

# Main Function
def main():
    # Initialize Swarm and Obstacles
    formation_radius = max(buffer_radius * num_robots / (2 * np.pi), buffer_radius)
    start_position = circle_center + circle_radius * np.array([1, 0])
    positions = initialize_positions(num_robots, start_position, formation_radius)
    headings = np.random.uniform(0, 2 * np.pi, num_robots)
    obstacles = generate_varied_obstacles_with_levels(circle_center, circle_radius, num_obstacles, min_obstacle_size, max_obstacle_size, offset_degrees, passage_width)

    # Set up the plot
    fig, ax = plt.subplots()
    scatter = ax.scatter(positions[:, 0], positions[:, 1], c='blue', label='Robots')
    # Plot Obstacles
    for obs in obstacles:
        if obs["type"] == "circle":
            circle = plt.Circle(obs["position"], obs["radius"], color='red', fill=True)
            ax.add_artist(circle)
    # Plot circular path
    circle = plt.Circle(circle_center, circle_radius, color='black', fill=False)
    ax.add_artist(circle)
    # Set up the plot
    ax.set_xlim(0, width)
    ax.set_ylim(0, width)
    ax.set_aspect('equal')

    # Animation
    ani = animation.FuncAnimation(
        fig, animate, frames=num_steps, interval=100, repeat=True,
        fargs=(positions, headings, formation_radius, obstacles, scatter)
    )
    plt.show()

if __name__ == "__main__":
    main()
