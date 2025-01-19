import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Import only the basics needed from ocm:
from ocm import (
    initialize_positions,
    initialize_positions_triangle,
    compute_forces_with_sensors,
    update_positions_and_headings,
    get_target_positions,
    get_target_positions_triangle,
    get_moving_center,
    # We do NOT import check_collisions here, because we won't remove robots in the env
    width, num_robots, formation_radius, max_speed, num_steps,
    generate_varied_obstacles_with_levels,
    num_obstacles, min_obstacle_size, max_obstacle_size, offset_degrees, passage_width, obstacle_level, K_base, C_base, formation_type, formation_size_triangle
)


class SwarmEnv(gym.Env):
    def __init__(self, seed_value=42, episode_length_factor=4):
        super(SwarmEnv, self).__init__()
        # Action space: Adjust K (alignment), C (cohesion)
        self.action_space = spaces.Box(
            low=np.array([0.005, 0.005]),
            high=np.array([0.995, 0.995]),
            dtype=np.float32
        )
        # Observation space shape (positions, headings, velocities)
        obs_shape = (num_robots * 5,)  # (x, y) position, heading, (x, y) velocity for each robot
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )
        
        self.num_obstacles = num_obstacles  # Number of obstacles
        self.min_obstacle_size = min_obstacle_size
        self.max_obstacle_size = max_obstacle_size
        self.offset_degrees = offset_degrees
        self.passage_width = passage_width
        self.obstacle_level = obstacle_level  # Choose the level of obstacles to combine different types
        self.formation_type = formation_type  # Choose the formation type

        self.width = width
        self.max_speed = max_speed
        self.seed(seed_value)

        # Initial “base” alignment & cohesion (overwritten by agent's actions)
        self.K_base = K_base
        self.C_base = C_base
        
        self.episode_length_factor = episode_length_factor

        self.reset()

    def seed(self, seed_value):
        """
        Seed the environment to ensure reproducibility.
        """
        np.random.seed(seed_value)
        self.seed_value = seed_value

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        # Center and radius for circular motion
        self.circle_center = np.array([self.width / 2, self.width / 2])
        self.circle_radius = self.width / 4

        # Generate obstacles
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

        # Initialize robot positions, headings, velocities
        # self.positions = initialize_positions(num_robots, self.circle_center, formation_radius)
        start_position = self.circle_center + self.circle_radius * np.array([1, 0])
        if self.formation_type == 'circle':
            self.positions = initialize_positions(num_robots, start_position, formation_radius)
        elif self.formation_type == 'triangle':
            self.positions = initialize_positions_triangle(num_robots, start_position, formation_size_triangle)
        # self.positions = initialize_positions(num_robots, start_position, formation_radius)
        self.headings = np.random.uniform(0, 2 * np.pi, num_robots)
        self.velocities = np.zeros_like(self.positions)

        self.steps = 0
        self.max_episode_steps = num_steps * self.episode_length_factor
        return self._get_obs(), {}

    def _get_obs(self):
        # Flatten positions, headings, velocities
        state = np.concatenate([
            self.positions.flatten(),
            self.headings.flatten(),
            self.velocities.flatten()
        ])
        return state
    
    def _calculate_average_pairwise_distance(self):
        """Calculate the average pairwise distance between all robots."""
        num_robots = len(self.positions)
        total_distance = 0
        count = 0

        for i in range(num_robots):
            for j in range(i + 1, num_robots):
                total_distance += np.linalg.norm(self.positions[i] - self.positions[j])
                count += 1

        if count == 0:  # Avoid division by zero
            return 0

        return total_distance / count

    def step(self, action):
        # 1. Take action = [K, C], override environment’s K_base, C_base
        self.K_base, self.C_base = action

        # 2. Compute next positions using forces
        moving_center = get_moving_center(self.steps, num_steps)
        if self.formation_type == 'circle':
            target_positions = get_target_positions(
                moving_center, len(self.positions), formation_radius
            )
        elif self.formation_type == 'triangle':
            target_positions = get_target_positions_triangle(
                moving_center, len(self.positions), formation_size_triangle
            )

        # We do NOT remove collided robots here.
        forces, _, _, _, collisions = compute_forces_with_sensors(
            self.positions,
            self.headings,
            self.velocities,
            target_positions,
            self.obstacles,
            self.K_base,
            self.C_base
        )

        # 3. Update swarm positions and headings
        self.positions, self.headings = update_positions_and_headings(
            self.positions,
            self.headings,
            forces,
            self.max_speed,
            (self.width, 0.5)
        )

        # 4. Simple reward (no collision removal). You can make this more sophisticated.
        reward = 1 # Base reward for each step
        done = False
        if collisions:  # Penalize collisions
            reward = -10 * len(collisions)
            done = True  # End episode if there are collisions
        if not done:
            if self.formation_type == 'circle':
                avg_distance = self._calculate_average_pairwise_distance()
                distance_threshold = formation_radius * 1  # Adjust threshold as needed
                if avg_distance < distance_threshold:
                    reward += 0.001  # Reward for staying close
                    
                if avg_distance > distance_threshold * 2.5:
                    reward -= 0.01
            elif self.formation_type == 'triangle':
                avg_distance = self._calculate_average_pairwise_distance()
                distance_threshold = formation_size_triangle * 1
                if avg_distance < distance_threshold:
                    reward += 0.0001
                if avg_distance > distance_threshold * 1.5:
                    reward -= 0.01

        self.steps += 1
        # done = self.steps >= self.max_episode_steps
        if self.steps >= self.max_episode_steps:
            done = True

        truncated = False  # No external truncation logic
        obs = self._get_obs()
        return obs, reward, done, truncated, {}

    def render(self, mode='rgb_array'):
        """
        If you want a quick way to visualize in offline playback:
        - For a real-time window, you can do matplotlib live.
        - Or return an RGB array if you want to collect frames for a video.

        Simple example: Return a blank image. (You could do a proper 2D draw here.)
        """
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        return image
