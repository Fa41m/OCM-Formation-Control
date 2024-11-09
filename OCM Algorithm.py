import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from PSO import pso

# Parameters
num_robots = 7      # Number of robots
num_steps = 800     # Number of time steps
alpha = 0.1         # Weight for repulsion force
K = 0.2             # Alignment strength
C = 0.05            # Cohesion strength
width = 35          # Width of the 2D space (world boundary)
buffer_radius = 0.3 # Minimum distance between robots
sensing_radius = 5.0 # Sensing radius for neighbors
constant_speed = 0.2 # Constant speed for all robots
num_checkpoints = 8 # Number of checkpoints (must be at least 2)
boundary_tolerance = 0.5 # Tolerance for boundary constraint
num_obstacles = 5   # Number of obstacles to spawn
obstacle_radius = 1.0 # Radius of obstacles

# Generate random checkpoints with distinct start and end positions
np.random.seed(21)  # For reproducibility
checkpoints = [np.random.rand(2) * width for _ in range(num_checkpoints)]

# Parameters for PSO
# TODO Hyperparameters for PSO
num_particles = 30  # Number of particles in the swarm
num_iterations = 100  # Number of PSO iterations
inertia = 0.5
cognitive_coeff = 1.5
social_coeff = 1.5

# Obtain optimized route using PSO
optimal_route = pso(checkpoints, num_particles, num_iterations, inertia, cognitive_coeff, social_coeff)

# Redefine start, intermediate, and end checkpoints based on optimized route
start_checkpoint = optimal_route[0]
intermediate_checkpoints = optimal_route[1:-1]
end_checkpoint = optimal_route[-1]

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

# Generate random obstacles
obstacles = [np.random.rand(2) * (width - 2 * obstacle_radius) + obstacle_radius for _ in range(num_obstacles)]

# Tracking variables for plotting
average_forces = []
average_alignments = []

# Calculate the moving center based on the frame number and optimized route
def get_moving_center(frame, num_steps, route):
    total_segments = len(route) - 1
    segment_length = num_steps // total_segments
    current_segment = min(frame // segment_length, total_segments - 1)
    t = (frame % segment_length) / segment_length
    start = np.array(route[current_segment])
    end = np.array(route[current_segment + 1])
    return (1 - t) * start + t * end

# Boundary condition enforcement
def enforce_boundary_conditions(positions, width, boundary_tolerance):
    corrected_positions = np.copy(positions)
    for i in range(len(positions)):
        for dim in range(2):  # Apply boundary conditions in x and y directions
            if corrected_positions[i][dim] < boundary_tolerance:
                corrected_positions[i][dim] = boundary_tolerance
            elif corrected_positions[i][dim] > width - boundary_tolerance:
                corrected_positions[i][dim] = width - boundary_tolerance
    return corrected_positions

# Calculate positions on the moving circle for each robot
def get_circle_positions(moving_center, radius, num_robots):
    angles = np.linspace(0, 2 * np.pi, num_robots, endpoint=False)
    return np.array([
        moving_center + radius * np.array([np.cos(angle), np.sin(angle)])
        for angle in angles
    ])

# Function to calculate repulsion, alignment, cohesion, and obstacle avoidance forces
def compute_forces(positions, headings, circle_positions):
    forces = np.zeros((num_robots, 2))
    total_force = 0  # Track total force for averaging
    
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

        # Obstacle avoidance force
        obstacle_avoidance_force = np.zeros(2)
        for obstacle in obstacles:
            distance_to_obstacle = np.linalg.norm(positions[i] - obstacle)
            if distance_to_obstacle < sensing_radius + obstacle_radius:
                # Apply repulsion from the obstacle based on distance
                obstacle_avoidance_force += (positions[i] - obstacle) / (distance_to_obstacle ** 2)

        # Total force calculation using control parameters
        forces[i] = (
            alpha * repulsion_force +
            C * cohesion_force +
            alpha * obstacle_avoidance_force  # Adjust the weight for obstacle avoidance
        )
        total_force += np.linalg.norm(forces[i])  # Sum the magnitude of forces

    average_force = total_force / num_robots
    return forces, average_force

# Calculate average alignment
def compute_alignment(headings):
    avg_heading = np.mean(headings)
    alignment = np.mean([np.cos(heading - avg_heading) for heading in headings])
    return alignment

# Update positions and headings based on forces
def update_positions_and_headings(positions, headings, forces):
    new_positions = np.copy(positions)
    new_headings = np.copy(headings)
    
    for i in range(num_robots):
        step_direction = np.array(forces[i]) / np.linalg.norm(forces[i]) if np.linalg.norm(forces[i]) != 0 else np.array([1, 0])
        new_positions[i] += constant_speed * step_direction
        new_headings[i] = np.arctan2(step_direction[1], step_direction[0])
    
    # Enforce boundary conditions
    new_positions = enforce_boundary_conditions(new_positions, width, boundary_tolerance)
    return new_positions, new_headings


# Set up the plot
fig, ax = plt.subplots()
scat = ax.scatter(positions[:, 0], positions[:, 1], c='blue', label='Robots')
scat_start = ax.scatter(start_checkpoint[0], start_checkpoint[1], c='green', marker='o', s=100, label='Start')
scat_end = ax.scatter(end_checkpoint[0], end_checkpoint[1], c='red', marker='x', s=100, label='End')
scat_intermediate = ax.scatter(*zip(*intermediate_checkpoints), c='orange', marker='s', s=60, label='Intermediate Checkpoints')
obstacle_patches = [plt.Circle((obs[0], obs[1]), obstacle_radius, color='gray', label='Obstacles') for obs in obstacles]

for patch in obstacle_patches:
    ax.add_patch(patch)

ax.set_xlim(0, width)
ax.set_ylim(0, width)
ax.set_title("Swarm Navigation through Optimized Route with Obstacles")
ax.legend()

# Animation update function
def animate(frame):
    global positions, headings
    moving_center = get_moving_center(frame, num_steps, optimal_route)
    radius = buffer_radius * 2.0 / np.sin(np.pi / num_robots)
    circle_positions = get_circle_positions(moving_center, radius, num_robots)
    
    forces, avg_force = compute_forces(positions, headings, circle_positions)
    alignment = compute_alignment(headings)
    
    # Store average force and alignment for plotting
    average_forces.append(avg_force)
    average_alignments.append(alignment)
    
    # Update positions and headings
    positions, headings = update_positions_and_headings(positions, headings, forces)
    
    # Update scatter plot data
    scat.set_offsets(positions)
    return scat, scat_start, scat_end, scat_intermediate

# Run the animation
ani = animation.FuncAnimation(fig, animate, frames=num_steps, interval=100, repeat=False)
plt.show()

# Plot the Average Force over Time
plt.figure()
plt.plot(average_forces)
plt.xlabel("Frame")
plt.ylabel("Average Force")
plt.title("Average Force between Robots over Time")
plt.show()  

# Plot the Alignment over Time
plt.figure()
plt.plot(average_alignments)
plt.xlabel("Frame")
plt.ylabel("Alignment")
plt.title("Average Alignment of Robots over Time")
plt.show()
