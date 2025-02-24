import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Import functions and parameters from your simulation module.
# (Here we assume you’ve renamed your simulation file to “new_ocm.py”)
from new_ocm import (
    initialize_positions,
    initialize_positions_triangle,
    compute_forces_with_sensors,
    update_positions_and_headings,
    get_target_positions,
    get_target_positions_triangle,
    get_moving_center,
    check_collisions,
    adapt_parameters,
    enforce_boundary_conditions,
    width,
    num_robots,
    formation_radius_base,
    formation_size_triangle_base,
    max_speed,
    num_steps,
    generate_varied_obstacles_with_levels,
    num_obstacles, min_obstacle_size, max_obstacle_size,
    offset_degrees, passage_width, obstacle_level,
    K_base, C_base, alpha_base, beta_base, formation_type,
    sensor_detection_distance,
    collision_zone  # used for collision checking
)

# In case you want to update velocities as well, you could create a helper:
def update_velocities(forces, max_speed):
    new_velocities = []
    for force in forces:
        speed = np.linalg.norm(force)
        if speed > max_speed:
            force = (max_speed / speed) * force
        new_velocities.append(force)
    return np.array(new_velocities)


class SwarmEnv(gym.Env):
    def __init__(self, seed_value=42, episode_length_factor=4):
        super(SwarmEnv, self).__init__()
        # Action space: agent controls the “base” alignment and cohesion parameters.
        self.action_space = spaces.Box(
            low=np.array([0.005, 0.005]),
            high=np.array([0.995, 0.995]),
            dtype=np.float32
        )
        # Observation: we include positions, headings, and velocities.
        obs_shape = (num_robots * 5,)  # for each robot: x, y, heading, vx, vy
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )
        
        # Obstacle and formation parameters:
        self.num_obstacles = num_obstacles
        self.min_obstacle_size = min_obstacle_size
        self.max_obstacle_size = max_obstacle_size
        self.offset_degrees = offset_degrees
        self.passage_width = passage_width
        self.obstacle_level = obstacle_level
        self.formation_type = formation_type  # "circle" or "triangle"

        self.width = width
        self.max_speed = max_speed
        self.seed(seed_value)

        # The “base” alignment & cohesion (will be overridden by the agent’s actions)
        self.K_base = K_base
        self.C_base = C_base
        # Also start with the base repulsion parameters
        self.alpha = alpha_base
        self.beta = beta_base
        
        self.episode_length_factor = episode_length_factor

        self.reset()

    def seed(self, seed_value):
        np.random.seed(seed_value)
        self.seed_value = seed_value

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        # Center and radius for the moving circle.
        self.circle_center = np.array([self.width / 2, self.width / 2])
        self.circle_radius = self.width / 4

        # Generate obstacles.
        self.obstacles = generate_varied_obstacles_with_levels(
            self.circle_center,
            self.circle_radius,
            num_obstacles,
            min_obstacle_size,
            max_obstacle_size,
            offset_degrees,
            passage_width,
            obstacle_level
        )

        # Initialize positions, headings, velocities.
        start_position = self.circle_center + self.circle_radius * np.array([1, 0])
        if self.formation_type.lower() == 'triangle':
            self.positions = initialize_positions_triangle(num_robots, start_position, formation_size_triangle_base)
        else:  # default "circle"
            self.positions = initialize_positions(num_robots, start_position, formation_radius_base)
        self.headings = np.random.uniform(0, 2 * np.pi, num_robots)
        self.velocities = np.zeros_like(self.positions)

        self.steps = 0
        self.max_episode_steps = num_steps * self.episode_length_factor
        return self._get_obs(), {}

    def _get_obs(self):
        # Flatten positions, headings, and velocities.
        state = np.concatenate([
            self.positions.flatten(),
            self.headings.flatten(),
            self.velocities.flatten()
        ])
        return state

    def _calculate_average_pairwise_distance(self):
        total_distance = 0.0
        count = 0
        for i in range(len(self.positions)):
            for j in range(i + 1, len(self.positions)):
                total_distance += np.linalg.norm(self.positions[i] - self.positions[j])
                count += 1
        return total_distance / count if count > 0 else 0

    def step(self, action):
        # 1. Override the environment’s K_base and C_base with the agent’s action.
        self.K_base, self.C_base = action

        # 2. Adapt the formation parameters and repulsion strengths based on the current positions.
        # (This returns the adapted formation radius/size and updates α and β.)
        adapted_formation_radius, adapted_formation_size_triangle, self.alpha, self.beta = adapt_parameters(
            self.positions, self.obstacles,
            formation_radius_base, formation_size_triangle_base,
            alpha_base, beta_base,
            min_dist_threshold=4.0
        )

        # 3. Compute the moving center and target positions.
        moving_center = get_moving_center(self.steps, num_steps)
        if self.formation_type.lower() == 'triangle':
            target_positions = get_target_positions_triangle(moving_center, len(self.positions), adapted_formation_size_triangle)
        else:
            target_positions = get_target_positions(moving_center, len(self.positions), adapted_formation_radius)

        # 4. Compute forces using the sensor-based function.
        forces, self.K_base, self.C_base = compute_forces_with_sensors(
            self.positions, self.headings, self.velocities,
            target_positions, self.obstacles,
            self.K_base, self.C_base,
            self.alpha, self.beta
        )

        # 5. Update positions and headings.
        self.positions, self.headings = update_positions_and_headings(
            self.positions, self.headings, forces, self.max_speed,
            (self.width, 0.5)
        )
        # Also update velocities (using the same force clipping as in the update function).
        self.velocities = update_velocities(forces, self.max_speed)

        # 6. Check for collisions without removing robots.
        # (Call the collision function on a copy so that the state is not modified.)
        _, collision_indices = check_collisions(np.copy(self.positions), self.obstacles)

        # 7. Reward calculation.
        # Start with a small positive reward per step.
        reward = 0.1
        done = False
        if collision_indices:
            reward = -50 * len(collision_indices)
            done = True  # End the episode if a collision occurs.
        else:
            # Reward is also based on how tightly the swarm is formed.
            avg_distance = self._calculate_average_pairwise_distance()
            if self.formation_type.lower() == 'circle':
                # Use the adapted formation radius as a threshold.
                distance_threshold = adapted_formation_radius * 1.0
                if avg_distance < distance_threshold:
                    reward += 1.0
                elif avg_distance > distance_threshold * 2.5:
                    reward -= 0.5
            elif self.formation_type.lower() == 'triangle':
                # For triangle formation, use the adapted triangle size.
                distance_threshold = adapted_formation_size_triangle * 1.0
                if avg_distance < distance_threshold:
                    reward += 0.005
                elif avg_distance > distance_threshold * 1.5:
                    reward -= 0.01
            # Clip reward if needed.
            reward = min(1000, reward)

        self.steps += 1
        if self.steps >= self.max_episode_steps:
            done = True

        truncated = False
        obs = self._get_obs()
        return obs, reward, done, truncated, {}

    def render(self, mode='rgb_array'):
        # A simple render that returns a blank image (or you could use matplotlib here).
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        return image
