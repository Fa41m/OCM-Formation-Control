import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
num_robots = 20      # Number of robots (adjustable)
num_steps = 400     # Number of time steps for a full circle
alpha = 0.1         # Weight for repulsion force
beta = 0.1          # Weight for attraction force
K = 0.2             # Alignment strength (initial)
C = 0.1             # Cohesion strength (initial)
width = 100          # Width of the 2D space (world boundary)
buffer_radius = 0.5 # Minimum distance between robots
sensing_radius = 0.5 # Sensing radius for neighbors
constant_speed = 0.1 # Base speed for all robots
max_speed = 0.3     # Maximum speed for robots
num_checkpoints = 10 # Number of checkpoints around the circle (unused in adjusted code)
boundary_tolerance = 0.5 # Tolerance for boundary constraint

# Parameters for varied obstacles
num_obstacles = 3  # Total number of obstacles
min_obstacle_size = 1.0
max_obstacle_size = 2.0
offset_degrees = 50
passage_width = 2.0  # Width of the passage for Level 3 obstacles

# Lists to log K and C values
K_values = []
C_values = []

# Define the center and radius of the circular path
circle_center = np.array([width / 2, width / 2])  # Center of the world
circle_radius = width / 4  # Radius of the circle

# Generate obstacles with three levels: offset, on the circle, and paired with a passage
def generate_varied_obstacles_with_levels(center, radius, num_obstacles, min_size, max_size, offset_degrees, passage_width):
    offset_radians = np.deg2rad(offset_degrees)  # Convert offset to radians
    angles = np.linspace(0, 2 * np.pi, num_obstacles, endpoint=False) + offset_radians  # Apply offset
    obstacles = []
    
    for i, angle in enumerate(angles):
        # Level 1: Offset obstacles
        if i % 3 == 0:  # For 1/3 of the obstacles
            offset_direction = np.random.choice([-1, 1])  # Randomly offset inward (-1) or outward (+1)
            offset_distance = 2 * offset_direction  # Offset by 2 units inward or outward
            offset_position = center + (radius + offset_distance) * np.array([np.cos(angle), np.sin(angle)])  # Along the radius
            size = np.random.uniform(min_size, max_size)
            obstacles.append({
                "type": "circle",
                "level": 1,
                "position": offset_position,
                "radius": size
            })

        # Level 2: Obstacles on the circle
        elif i % 3 == 1:  # For 1/3 of the obstacles
            position = center + radius * np.array([np.cos(angle), np.sin(angle)])
            size = np.random.uniform(min_size, max_size)
            shape = np.random.choice(["circle"])
            if shape == "circle":
                obstacles.append({
                    "type": "circle",
                    "level": 2,
                    "position": position,
                    "radius": size
                })

        # Level 3: Paired obstacles with a passage
        elif i % 3 == 2:
            # Randomly determine radii for the two obstacles
            size1 = np.random.uniform(min_size, max_size)
            size2 = np.random.uniform(min_size, max_size)
            
            # Ensure the passage width is at least 1.5 units
            adjusted_passage_width = max(passage_width, size1 + size2 + 1.5)
            
            # Calculate the point on the circle where the radius intersects
            central_position = center + radius * np.array([np.cos(angle), np.sin(angle)])
            
            # Position the two obstacles along the radius (one inside, one outside)
            position1 = central_position - np.array([np.cos(angle), np.sin(angle)]) * (adjusted_passage_width / 2)
            position2 = central_position + np.array([np.cos(angle), np.sin(angle)]) * (adjusted_passage_width / 2)
            
            # Add the two obstacles to the list
            obstacles.append({
                "type": "circle",
                "level": 3,
                "position": position1,
                "radius": size1
            })
            obstacles.append({
                "type": "circle",
                "level": 3,
                "position": position2,
                "radius": size2
            })

    return obstacles

# Generate the obstacles with three levels
obstacles = generate_varied_obstacles_with_levels(circle_center, circle_radius, num_obstacles, min_obstacle_size, max_obstacle_size, offset_degrees, passage_width)

# Initialize positions in a triangle shape around the start point on the circle
def initialize_positions(num_robots, start_position, formation_size):
    # Compute heading angle normal to the circle at start_position
    radius_vector = start_position - circle_center
    heading_vector = radius_vector  # Normal vector
    heading_angle = np.arctan2(heading_vector[1], heading_vector[0])

    # Determine the number of rows needed for the triangle formation
    row = 1
    total_bots = 0
    rows = []
    while total_bots < num_robots:
        bots_in_row = row
        if total_bots + bots_in_row > num_robots:
            bots_in_row = num_robots - total_bots  # Adjust for incomplete last row
        rows.append(bots_in_row)
        total_bots += bots_in_row
        row += 1

    # Generate positions relative to the formation center
    positions_relative = []
    h_spacing = formation_size * 1.5  # Vertical spacing between rows
    v_spacing = formation_size        # Horizontal spacing between robots in a row
    y_offset = 0

    for i, bots_in_row in enumerate(rows):
        x_offset = -v_spacing * (bots_in_row - 1) / 2  # Center the row
        for j in range(bots_in_row):
            positions_relative.append([x_offset + j * v_spacing, y_offset])
        y_offset -= h_spacing  # Move to the next row

    positions_relative = np.array(positions_relative)

    # Rotate positions by heading_angle
    rotation_matrix = np.array([
        [np.cos(heading_angle), -np.sin(heading_angle)],
        [np.sin(heading_angle),  np.cos(heading_angle)]
    ])

    rotated_positions = positions_relative @ rotation_matrix.T

    # Translate positions to start_position
    initial_positions = rotated_positions + start_position

    return initial_positions

# Initial start position on the circle
start_angle = 0
start_position = circle_center + circle_radius * np.array([np.cos(start_angle), np.sin(start_angle)])
positions = initialize_positions(num_robots, start_position, buffer_radius)
headings = np.random.rand(num_robots) * 2 * np.pi  # Random initial headings

# Adjusted get_moving_center function
def get_moving_center(frame, total_frames):
    # Compute the angle corresponding to the current frame
    theta = 2 * np.pi * frame / total_frames
    # Compute the moving_center along the circle
    moving_center = circle_center + circle_radius * np.array([np.cos(theta), np.sin(theta)])
    return moving_center

# Adjusted get_target_positions function
def get_target_positions(moving_center, num_robots, formation_size):
    # Compute heading angle normal to the circle at moving_center
    radius_vector = moving_center - circle_center
    heading_vector = radius_vector  # Normal vector
    heading_angle = np.arctan2(heading_vector[1], heading_vector[0])

    # Determine the number of rows needed for the triangle formation
    row = 1
    total_bots = 0
    rows = []
    while total_bots < num_robots:
        bots_in_row = row
        if total_bots + bots_in_row > num_robots:
            bots_in_row = num_robots - total_bots  # Adjust for incomplete last row
        rows.append(bots_in_row)
        total_bots += bots_in_row
        row += 1

    # Generate positions relative to the formation center
    positions_relative = []
    h_spacing = formation_size * 1.5  # Vertical spacing between rows
    v_spacing = formation_size        # Horizontal spacing between robots in a row
    y_offset = 0

    for i, bots_in_row in enumerate(rows):
        x_offset = -v_spacing * (bots_in_row - 1) / 2  # Center the row
        for j in range(bots_in_row):
            positions_relative.append([x_offset + j * v_spacing, y_offset])
        y_offset -= h_spacing  # Move to the next row

    positions_relative = np.array(positions_relative)

    # Rotate positions by heading_angle
    rotation_matrix = np.array([
        [np.cos(heading_angle), -np.sin(heading_angle)],
        [np.sin(heading_angle),  np.cos(heading_angle)]
    ])

    rotated_positions = positions_relative @ rotation_matrix.T

    # Translate positions to moving_center
    target_positions = rotated_positions + moving_center

    return target_positions

# Adjust Beta dynamically based on swarm behavior or distance to moving center
def adjust_beta(positions, moving_center):
    global beta
    distances_to_center = np.linalg.norm(positions - moving_center, axis=1)
    avg_distance = np.mean(distances_to_center)
    desired_distance = 0.0  # Desired average distance to moving center
    error = avg_distance - desired_distance

    # Proportional gain (tunable parameter)
    Kp_beta = 0.1
    beta_change = Kp_beta * error
    beta = np.clip(beta + beta_change, 0.05, 0.5)  # Ensure beta stays within bounds

# Dynamically adjust K and C based on robot states
def adjust_parameters(positions, headings, target_positions):
    global K, C

    # Calculate average distance from target positions
    distances_to_targets = np.linalg.norm(positions - target_positions, axis=1)
    avg_distance_to_targets = np.mean(distances_to_targets)
    desired_distance = 0.0  # Ideally, robots are at their target positions
    error_distance = avg_distance_to_targets - desired_distance

    # Calculate alignment error
    alignment = compute_alignment(headings)
    desired_alignment = 1.0  # Perfect alignment
    error_alignment = desired_alignment - alignment  # Higher error means less alignment

    # Proportional gains (tunable parameters)
    Kp_C = 0.1
    Kp_K = 0.1

    # Adjust C (cohesion strength)
    C_change = Kp_C * error_distance
    C = np.clip(C + C_change, 0.05, 0.95)

    # Adjust K (alignment strength)
    K_change = Kp_K * error_alignment
    K = np.clip(K + K_change, 0.05, 0.95)

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

# Adjusted compute_forces function
def compute_forces(positions, headings, target_positions, moving_center, obstacles):
    forces = np.zeros((num_robots, 2))
    total_force = 0  # Track total force for averaging

    for i in range(num_robots):
        neighbors = [
            j for j in range(num_robots)
            if i != j and np.linalg.norm(positions[i] - positions[j]) < sensing_radius
        ]

        # Repulsion force from other robots
        repulsion_force = np.zeros(2)
        for j in neighbors:
            displacement = positions[i] - positions[j]
            distance = np.linalg.norm(displacement)
            direction = displacement / (distance + 1e-6)
            if distance < buffer_radius:
                # Strong repulsion when too close
                repulsion_force += direction * (1 / (distance + 1e-6) - 1 / buffer_radius) * (1 / (distance + 1e-6))
            else:
                # Weaker repulsion when within sensing radius
                repulsion_force += direction * (1 / (distance ** 2 + 1e-6))

        # Cohesion force towards the target position on the formation
        cohesion_force = target_positions[i] - positions[i]

        # Alignment force
        avg_heading = np.mean([headings[j] for j in neighbors]) if neighbors else headings[i]
        alignment_force = np.array([np.cos(avg_heading), np.sin(avg_heading)]) - np.array([np.cos(headings[i]), np.sin(headings[i])])

        # Obstacle avoidance force and decision-making
        obstacle_avoidance_force = np.zeros(2)
        for obs in obstacles:
            obs_position = obs["position"]
            obs_radius = obs["radius"] if obs["type"] == "circle" else max(obs["width"], obs["height"]) / 2
            distance_to_obstacle = np.linalg.norm(positions[i] - obs_position)

            if distance_to_obstacle < obs_radius + buffer_radius:  # Collision course with buffer zone
                # Compute tangent vectors for left and right avoidance
                radius_vector = positions[i] - obs_position
                tangent_left = np.array([-radius_vector[1], radius_vector[0]])  # Perpendicular left
                tangent_right = np.array([radius_vector[1], -radius_vector[0]])  # Perpendicular right

                # Normalize tangents
                tangent_left /= np.linalg.norm(tangent_left) + 1e-6
                tangent_right /= np.linalg.norm(tangent_right) + 1e-6

                # Decide left or right based on surrounding space
                left_clearance = np.linalg.norm((positions[i] + tangent_left * 0.5) - obs_position)
                right_clearance = np.linalg.norm((positions[i] + tangent_right * 0.5) - obs_position)

                if left_clearance > right_clearance:  # Right is clearer
                    obstacle_avoidance_force += tangent_right * 10.0  # Strong weight for avoidance
                else:  # Left is clearer
                    obstacle_avoidance_force += tangent_left * 10.0  # Strong weight for avoidance

        # Attraction force towards the moving center (for all robots)
        attraction_force = beta * (moving_center - positions[i])

        # Total force calculation
        if np.linalg.norm(obstacle_avoidance_force) > 0:  # If avoidance is active
            forces[i] = obstacle_avoidance_force  # Obstacle avoidance takes precedence
        else:
            forces[i] = (
                alpha * repulsion_force +
                C * cohesion_force +
                K * alignment_force +
                attraction_force
            )

        total_force += np.linalg.norm(forces[i])  # Sum the magnitude of forces

    average_force = total_force / num_robots
    return forces, average_force

# Update positions and headings based on forces and dynamic velocity
def update_positions_and_headings(positions, headings, forces, target_positions):
    new_positions = np.copy(positions)
    new_headings = np.copy(headings)
    
    for i in range(num_robots):
        if np.linalg.norm(forces[i]) != 0:
            step_direction = forces[i] / np.linalg.norm(forces[i])
        else:
            step_direction = np.array([1, 0])
        
        # Calculate velocity based on distance to target position
        distance_to_target = np.linalg.norm(target_positions[i] - positions[i])
        velocity = min(max_speed, constant_speed + 0.1 * distance_to_target)  # Dynamic velocity
        
        new_positions[i] += velocity * step_direction
        new_headings[i] = np.arctan2(step_direction[1], step_direction[0])
    
    # Enforce boundary conditions
    new_positions = enforce_boundary_conditions(new_positions, width, boundary_tolerance)
    
    # Ensure minimum separation between robots
    for i in range(num_robots):
        for j in range(i + 1, num_robots):
            displacement = new_positions[i] - new_positions[j]
            distance = np.linalg.norm(displacement)
            if distance < buffer_radius:
                # Adjust positions to maintain minimum separation
                overlap = buffer_radius - distance
                correction = (overlap / 2) * (displacement / (distance + 1e-6))
                new_positions[i] += correction
                new_positions[j] -= correction
    
    return new_positions, new_headings


# Set up the plot
fig, ax = plt.subplots()
scat = ax.scatter(positions[:, 0], positions[:, 1], c='blue', label='Swarm')
# Plot the initial moving center
# moving_center = get_moving_center(0, num_steps)
# scat_center = ax.scatter(moving_center[0], moving_center[1], c='orange', marker='o', label='Moving Center')
# Plot obstacles
for obs in obstacles:
    if obs["type"] == "circle":
        circle = plt.Circle(obs["position"], obs["radius"], color='green', alpha=0.5)
        ax.add_artist(circle)

# Draw the circle path
theta = np.linspace(0, 2 * np.pi, 100)
circle_path = circle_center[:, None] + circle_radius * np.array([np.cos(theta), np.sin(theta)])
ax.plot(circle_path[0, :], circle_path[1, :], linestyle='--', color='gray', label='Circle Path')

ax.set_xlim(0, width)
ax.set_ylim(0, width)
ax.set_aspect('equal')  # Set aspect ratio to 'equal' for accurate representation
ax.set_title("Swarm Following Circular Trajectory with Dynamic Triangle Formation")
ax.legend()

# Animation update function
def animate(frame):
    global positions, headings, K_values, C_values, beta
    total_frames = num_steps  # Total number of frames for a full circle

    # Get the moving center along the circle
    moving_center = get_moving_center(frame, total_frames)

    # Target positions around the moving center
    target_positions = get_target_positions(moving_center, num_robots, buffer_radius)

    # Adjust Beta dynamically based on swarm behavior
    adjust_beta(positions, moving_center)

    # Adjust K and C dynamically
    adjust_parameters(positions, headings, target_positions)

    # Log K and C values
    K_values.append(K)
    C_values.append(C)

    # Compute forces and update positions (including obstacle avoidance)
    forces, avg_force = compute_forces(positions, headings, target_positions, moving_center, obstacles)
    alignment = compute_alignment(headings)

    # Update positions and headings
    positions, headings = update_positions_and_headings(positions, headings, forces, target_positions)

    # Update scatter plot data
    scat.set_offsets(positions)
    # scat_center.set_offsets(moving_center)

    # return scat, scat_center
    return scat

# Run the animation indefinitely
ani = animation.FuncAnimation(fig, animate, frames=num_steps, interval=100, repeat=True)
plt.show()

# Plot K and C over time
plt.figure(figsize=(10, 5))
plt.plot(K_values, label="K (Alignment Strength)")
plt.plot(C_values, label="C (Cohesion Strength)")
plt.xlabel("Frame")
plt.ylabel("Value")
plt.title("Dynamic Adjustment of K and C Over Time")
plt.legend()
plt.grid(True)
plt.show()
