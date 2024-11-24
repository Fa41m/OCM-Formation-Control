import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
num_robots = 5       # Number of robots
num_steps = 800      # Number of time steps
alpha = 0.2          # Weight for repulsion force
K = 0.2              # Alignment strength
C = 0.1              # Cohesion strength
width = 35           # Width of the 2D space (world boundary)
buffer_radius = 0.5  # Minimum distance between robots
sensing_radius = 7.5 # Sensing radius for neighbors
constant_speed = 0.2 # Constant speed for all robots
num_checkpoints = 10  # Number of checkpoints around the circle
boundary_tolerance = 0.5 # Tolerance for boundary constraint

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
start_checkpoint = checkpoints[0]
rest_checkpoints = checkpoints[1:]

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

# Tracking variables for plotting
average_forces = []
average_alignments = []

# Calculate the moving center based on the current frame and circular checkpoints
def get_moving_center(frame, num_steps, checkpoints):
    total_segments = len(checkpoints)
    segment_length = num_steps // total_segments
    current_segment = min(frame // segment_length, total_segments - 1)
    t = (frame % segment_length) / segment_length
    start = np.array(checkpoints[current_segment])
    end = np.array(checkpoints[(current_segment + 1) % total_segments])
    return (1 - t) * start + t * end

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

# Compute forces based on OCM principles
def compute_forces(positions, headings, moving_center):
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

        # Cohesion force towards the moving center
        cohesion_force = moving_center - positions[i]

        # Alignment force
        avg_heading = np.mean([headings[j] for j in neighbors]) if neighbors else headings[i]
        alignment_force = np.array([np.cos(avg_heading), np.sin(avg_heading)]) - np.array([np.cos(headings[i]), np.sin(headings[i])])

        # Total force calculation using OCM principles
        forces[i] = (
            alpha * repulsion_force +
            C * cohesion_force +
            K * alignment_force
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
scat = ax.scatter(positions[:, 0], positions[:, 1], c='blue', label='Swarm')
scat_checkpoints = ax.scatter(checkpoints[:, 0], checkpoints[:, 1], c='red', marker='x', label='Checkpoints')

ax.set_xlim(0, width)
ax.set_ylim(0, width)
ax.set_aspect('equal')  # Set aspect ratio to 'equal' for accurate representation
ax.set_title("Swarm Following Circular Checkpoints")
ax.legend()

# Animation update function
def animate(frame):
    global positions, headings
    moving_center = get_moving_center(frame, num_steps, checkpoints)
    
    # Compute forces and update positions
    forces, avg_force = compute_forces(positions, headings, moving_center)
    alignment = compute_alignment(headings)
    
    # Store average force and alignment for plotting
    average_forces.append(avg_force)
    average_alignments.append(alignment)
    
    # Update positions and headings
    positions, headings = update_positions_and_headings(positions, headings, forces)
    
    # Update scatter plot data
    scat.set_offsets(positions)
    return scat,

# Run the animation
ani = animation.FuncAnimation(fig, animate, frames=num_steps, interval=100, repeat=True)
plt.show()
