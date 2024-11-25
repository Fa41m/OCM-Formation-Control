import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
num_robots = 5       # Number of robots
num_steps = 800      # Number of time steps
alpha = 0.1          # Weight for repulsion force
beta = 0.1           # Weight for attraction force
K = 0.2              # Alignment strength (initial)
C = 0.1              # Cohesion strength (initial)
width = 35           # Width of the 2D space (world boundary)
buffer_radius = 0.5  # Minimum distance between robots
sensing_radius = 7.5 # Sensing radius for neighbors
constant_speed = 0.1 # Base speed for all robots
max_speed = 0.3      # Maximum speed for robots
num_checkpoints = 10 # Number of checkpoints around the circle
boundary_tolerance = 0.5 # Tolerance for boundary constraint

# Lists to log K and C values
K_values = []
C_values = []

# Define the center and radius of the circular path
circle_center = np.array([width / 2, width / 2])  # Center of the world
circle_radius = width / 4  # Radius of the circle

# Generate circular checkpoints
def generate_circular_checkpoints(center, radius, num_checkpoints):
    angles = np.linspace(0, 2 * np.pi, num_checkpoints, endpoint=False)
    return np.array([
        center + radius * np.array([np.cos(angle), np.sin(angle)])
        for angle in angles
    ])

checkpoints = generate_circular_checkpoints(circle_center, circle_radius, num_checkpoints)

# Initialize positions in a circular shape around the start checkpoint
def initialize_positions(num_robots, start_position, buffer_radius):
    angles = np.linspace(0, 2 * np.pi, num_robots, endpoint=False)
    positions = np.array([
        start_position + buffer_radius * np.array([np.cos(angle), np.sin(angle)])
        for angle in angles
    ])
    return positions

positions = initialize_positions(num_robots, checkpoints[0], buffer_radius)
headings = np.random.rand(num_robots) * 2 * np.pi  # Random initial headings

# Calculate the moving center based on the current frame and circular checkpoints
def get_moving_center(frame, num_steps, checkpoints):
    total_segments = len(checkpoints)
    segment_length = num_steps // total_segments
    current_segment = min(frame // segment_length, total_segments - 1)
    t = (frame % segment_length) / segment_length
    start = np.array(checkpoints[current_segment])
    end = np.array(checkpoints[(current_segment + 1) % total_segments])
    return (1 - t) * start + t * end

# Calculate positions on a circular formation around the moving center
def get_target_positions(moving_center, num_robots, formation_radius):
    angles = np.linspace(0, 2 * np.pi, num_robots, endpoint=False)
    return np.array([
        moving_center + formation_radius * np.array([np.cos(angle), np.sin(angle)])
        for angle in angles
    ])

# Adjust Beta dynamically based on swarm behavior or distance to checkpoint
def adjust_beta(positions, current_checkpoint):
    global beta
    distances_to_checkpoint = np.linalg.norm(positions - current_checkpoint, axis=1)
    avg_distance = np.mean(distances_to_checkpoint)
    
    if avg_distance > 5.0:  # Example condition: Swarm is far from the checkpoint
        beta = min(0.5, beta + 0.01)  # Gradually increase beta
    else:
        beta = max(0.05, beta - 0.01)  # Gradually decrease beta

# Dynamically adjust K and C based on robot states
def adjust_parameters(positions, headings, target_positions):
    global K, C
    
    # Calculate average distance from target positions
    distances_to_targets = np.linalg.norm(positions - target_positions, axis=1)
    avg_distance_to_targets = np.mean(distances_to_targets)
    
    # Calculate alignment error
    alignment = compute_alignment(headings)
    alignment_error = 1 - alignment  # Higher error means less alignment
    
    # Adjust C (cohesion strength)
    if avg_distance_to_targets > 1.0:  # If robots are far from targets
        C = min(0.7, C + 0.01)  # Gradually increase cohesion strength
    else:
        C = max(0.05, C - 0.01)  # Gradually decrease cohesion strength
    
    # Adjust K (alignment strength)
    if alignment_error > 0.2:  # If alignment is poor
        K = min(0.7, K + 0.01)  # Gradually increase alignment strength
    else:
        K = max(0.05, K - 0.01)  # Gradually decrease alignment strength

# Compute alignment metric
def compute_alignment(headings):
    avg_heading = np.mean([np.exp(1j * heading) for heading in headings])
    alignment = np.abs(avg_heading)  # The magnitude of the mean heading as alignment metric
    return alignment

# Enforce boundary conditions
def enforce_boundary_conditions(positions, width, boundary_tolerance):
    corrected_positions = np.copy(positions)
    for i in range(len(positions)):
        for dim in range(2):  # Apply boundary conditions in x and y directions
            if corrected_positions[i][dim] < boundary_tolerance:
                corrected_positions[i][dim] = boundary_tolerance
            elif corrected_positions[i][dim] > width - boundary_tolerance:
                corrected_positions[i][dim] = width - boundary_tolerance
    return corrected_positions

# Compute forces based on OCM principles with Beta
def compute_forces_with_beta(positions, headings, target_positions, next_checkpoint):
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

        # Cohesion force towards the target position on the formation
        cohesion_force = target_positions[i] - positions[i]

        # Attraction force towards the next checkpoint
        attraction_force = next_checkpoint - positions[i]

        # Alignment force
        avg_heading = np.mean([headings[j] for j in neighbors]) if neighbors else headings[i]
        alignment_force = np.array([np.cos(avg_heading), np.sin(avg_heading)]) - np.array([np.cos(headings[i]), np.sin(headings[i])])

        # Total force calculation using OCM principles
        forces[i] = (
            alpha * repulsion_force +
            C * cohesion_force +
            K * alignment_force +
            beta * attraction_force  # Add beta for attraction force
        )
        total_force += np.linalg.norm(forces[i])  # Sum the magnitude of forces

    average_force = total_force / num_robots
    return forces, average_force

# Update positions and headings based on forces and dynamic velocity
def update_positions_and_headings(positions, headings, forces, target_positions):
    new_positions = np.copy(positions)
    new_headings = np.copy(headings)
    
    for i in range(num_robots):
        step_direction = np.array(forces[i]) / np.linalg.norm(forces[i]) if np.linalg.norm(forces[i]) != 0 else np.array([1, 0])
        
        # Calculate velocity based on distance to target position
        distance_to_target = np.linalg.norm(target_positions[i] - positions[i])
        velocity = min(max_speed, constant_speed + 0.1 * distance_to_target)  # Dynamic velocity
        
        new_positions[i] += velocity * step_direction
        new_headings[i] = np.arctan2(step_direction[1], step_direction[0])
    
    # Enforce boundary conditions
    new_positions = enforce_boundary_conditions(new_positions, width, boundary_tolerance)
    return new_positions, new_headings

# Set up the plot
fig, ax = plt.subplots()
scat = ax.scatter(positions[:, 0], positions[:, 1], c='blue', label='Swarm')
scat_checkpoints = ax.scatter(checkpoints[:, 0], checkpoints[:, 1], c='red', marker='x', label='Checkpoints')

ax.set_xlim(0, width)
ax.set_ylim(0, width)
ax.set_aspect('equal')  # Set aspect ratio to 'equal' for accurate representation
ax.set_title("Swarm Following Circular Checkpoints with Dynamic K, C, Beta, and Velocity")
ax.legend()

# Animation update function
# Animation update function
def animate(frame):
    global positions, headings, K_values, C_values
    # Calculate the total number of frames per checkpoint cycle
    frames_per_checkpoint = num_steps // num_checkpoints

    # Wrap frame to loop through checkpoints
    current_segment = (frame // frames_per_checkpoint) % num_checkpoints
    next_segment = (current_segment + 1) % num_checkpoints
    
    # Calculate interpolation factor for smooth transition between checkpoints
    t = (frame % frames_per_checkpoint) / frames_per_checkpoint
    current_checkpoint = checkpoints[current_segment]
    next_checkpoint = checkpoints[next_segment]
    moving_center = (1 - t) * current_checkpoint + t * next_checkpoint
    
    # Target positions around the moving center
    target_positions = get_target_positions(moving_center, num_robots, buffer_radius)
    
    # Adjust Beta dynamically based on swarm behavior
    adjust_beta(positions, current_checkpoint)
    
    # Adjust K and C dynamically
    adjust_parameters(positions, headings, target_positions)
    
    # Log K and C values
    K_values.append(K)
    C_values.append(C)
    
    # Compute forces and update positions
    forces, avg_force = compute_forces_with_beta(positions, headings, target_positions, next_checkpoint)
    alignment = compute_alignment(headings)
    
    # Update positions and headings
    positions, headings = update_positions_and_headings(positions, headings, forces, target_positions)
    
    # Update scatter plot data
    scat.set_offsets(positions)
    return scat,

# Run the animation indefinitely
ani = animation.FuncAnimation(fig, animate, interval=100, repeat=True)
plt.show()

# Plot K and C over time
plt.figure(figsize=(10, 5))
plt.plot(K_values, label="K (Alignment Strength)")
plt.plot(C_values, label="C (Cohesion Strength)")
plt.xlabel("Frame")
plt.ylabel("Value")
plt.title("Dynamic Adjustment of K, C, and Beta Over Time")
plt.legend()
plt.grid(True)
plt.show()
