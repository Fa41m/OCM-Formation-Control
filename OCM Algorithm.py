import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
num_robots = 7      # Number of robots
num_steps = 300     # Number of time steps
alpha = 0.1         # Weight for repulsion force
K = 0.2             # Alignment strength
C = 0.05            # Cohesion strength
width = 20          # Width of the 2D space
buffer_radius = 0.5 # Minimum distance between robots
sensing_radius = 5.0 # Sensing radius for neighbors
constant_speed = 0.2 # Constant speed for all robots
num_checkpoints = 5 # Number of checkpoints (must be at least 2)

# Generate random checkpoints with distinct start and end positions
np.random.seed(21)  # For reproducibility
checkpoints = [np.random.rand(2) * width for _ in range(num_checkpoints)]
start_checkpoint = checkpoints[0]
end_checkpoint = checkpoints[-1]

# Initialize positions in a circular shape around the start checkpoint
def initialize_positions(num_robots, start_position, buffer_radius):
    radius = buffer_radius * 2.0 / np.sin(np.pi / num_robots)
    angles = np.linspace(0, 2 * np.pi, num_robots, endpoint=False)
    positions = np.array([
        start_position + radius * np.array([np.cos(angle), np.sin(angle)])
        for angle in angles
    ])
    return positions

positions = initialize_positions(num_robots, start_checkpoint, buffer_radius)
headings = np.random.rand(num_robots) * 2 * np.pi  # Random initial headings

# Calculate the moving center based on the frame number and current checkpoint
def get_moving_center(frame, num_steps, checkpoints):
    total_segments = len(checkpoints) - 1
    segment_length = num_steps // total_segments
    current_segment = min(frame // segment_length, total_segments - 1)
    t = (frame % segment_length) / segment_length
    return (1 - t) * checkpoints[current_segment] + t * checkpoints[current_segment + 1]

# Calculate positions on the moving circle for each robot
def get_circle_positions(moving_center, radius, num_robots):
    angles = np.linspace(0, 2 * np.pi, num_robots, endpoint=False)
    return np.array([
        moving_center + radius * np.array([np.cos(angle), np.sin(angle)])
        for angle in angles
    ])

# Function to calculate repulsion, alignment, and cohesion forces
def compute_forces(positions, headings, circle_positions):
    forces = np.zeros((num_robots, 2))
    
    for i in range(num_robots):
        neighbors = [
            j for j in range(num_robots)
            if i != j and np.linalg.norm(positions[i] - positions[j]) < sensing_radius
        ]

        # Repulsion force
        repulsion_force = np.zeros(2)
        for j in neighbors:
            distance = np.linalg.norm(positions[i] - positions[j])
            if distance < buffer_radius:
                repulsion_force += (positions[i] - positions[j]) / (distance ** 2)

        # Cohesion force towards the designated position on the moving circle
        cohesion_force = circle_positions[i] - positions[i]

        # Total force calculation using control parameters
        forces[i] = (
            alpha * repulsion_force +
            C * cohesion_force
        )

    return forces

# Update positions and headings based on forces
def update_positions_and_headings(positions, headings, forces):
    new_positions = np.copy(positions)
    new_headings = np.copy(headings)
    
    for i in range(num_robots):
        # Ensure step_direction is a numpy array
        step_direction = np.array(forces[i]) / np.linalg.norm(forces[i]) if np.linalg.norm(forces[i]) != 0 else np.array([1, 0])
        new_positions[i] += constant_speed * step_direction
        new_headings[i] = np.arctan2(step_direction[1], step_direction[0])
    
    return new_positions, new_headings

# Set up the plot
fig, ax = plt.subplots()
scat = ax.scatter(positions[:, 0], positions[:, 1], c='blue', label='Robots')
scat_start = ax.scatter(start_checkpoint[0], start_checkpoint[1], c='green', marker='o', s=100, label='Start')
scat_end = ax.scatter(end_checkpoint[0], end_checkpoint[1], c='red', marker='x', s=100, label='End')
intermediate_checkpoints = checkpoints[1:-1]
scat_intermediate = ax.scatter(*zip(*intermediate_checkpoints), c='orange', marker='s', s=60, label='Intermediate Checkpoints')

ax.set_xlim(0, width)
ax.set_ylim(0, width)
ax.set_title("Swarm Navigation through Multiple Checkpoints")
ax.legend()

# Animation update function
def animate(frame):
    global positions, headings
    moving_center = get_moving_center(frame, num_steps, checkpoints)
    radius = buffer_radius * 2.0 / np.sin(np.pi / num_robots)
    circle_positions = get_circle_positions(moving_center, radius, num_robots)
    
    forces = compute_forces(positions, headings, circle_positions)
    positions, headings = update_positions_and_headings(positions, headings, forces)
    
    # Update scatter plot data
    scat.set_offsets(positions)
    return scat, scat_start, scat_end, scat_intermediate

# Run the animation
ani = animation.FuncAnimation(fig, animate, frames=num_steps, interval=100, repeat=False)
plt.show()
