import gymnasium as gym
import numpy as np

from new_ocm import (
    initialize_positions,
    initialize_positions_triangle,
    compute_forces_with_sensors,
    update_positions_and_headings,
    # adapt_parameters  # We are no longer calling this, so can be ignored/removed
    check_collisions,
    enforce_boundary_conditions,
    # Various global constants
    world_width,
    num_robots,
    robot_max_speed,
    num_steps,
    generate_varied_obstacles_with_levels,
    num_obstacles, min_obstacle_size, max_obstacle_size,
    offset_degrees, passage_width, obstacle_level,
    cost_w1, cost_w2, cost_w3, psi_threshold,
    compute_swarm_alignment,
    formation_type,
    formation_radius_base,
    formation_size_triangle_base,
    sensor_detection_distance,
    # We used to rely on alpha_base, etc., but now RL overrides them entirely
)

def update_velocities(forces, robot_max_speed):
    # Same as before: clip forces by speed
    clipped = []
    for f in forces:
        spd = np.linalg.norm(f)
        if spd > robot_max_speed:
            f = (robot_max_speed / spd) * f
        clipped.append(f)
    return np.array(clipped)

class SwarmEnv(gym.Env):
    """
    Example environment implementing:
      (1) Heavier collision penalty & survival bonus
      (2) RL control over alpha,beta,K,C (but NOT formation_radius anymore)
      (3) Path completion (4 laps) bonus + partial progress reward
    """

    def __init__(self, seed_value=42):
        super().__init__()

        # -------------------------------
        # Action space:
        # Now 4D => [alpha, beta, K, C]
        # alpha,beta in [0,1], K,C in [0.005,0.995], etc.
        # Formation radius is fixed internally, not learned.
        # -------------------------------
        self.formation_type = formation_type
        if formation_type.lower() == 'triangle':
            self.formation_base = formation_size_triangle_base
        else:
            self.formation_base = formation_radius_base

        self.action_space = gym.spaces.Box(
            low=np.array([0.0,  0.0,   0.005, 0.005]),
            high=np.array([1.0, 1.0,   0.995, 0.995]),
            dtype=np.float32
        )

        # Observation space: same as before
        obs_dim = num_robots * 5  # (x,y,heading,vx,vy) each
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )

        self.world_width = world_width
        self.robot_max_speed = robot_max_speed

        self.num_obstacles = num_obstacles
        self.obstacle_level = obstacle_level

        # 4 laps => 8π
        self.max_angle = 8.0 * np.pi
        self.current_angle_accum = 0.0

        self.steps = 0
        # for partial progress reward
        self.last_angle_accum = 0.0

        self.seed(seed_value)
        self._setup_episode()

    def seed(self, seed_val):
        np.random.seed(seed_val)
        self.seed_value = seed_val

    def _setup_episode(self):
        # Circle center & radius
        self.circle_center = np.array([self.world_width / 2, self.world_width / 2])
        self.circle_radius = self.world_width / 4

        # Obstacles
        self.obstacles = generate_varied_obstacles_with_levels(
            center=self.circle_center,
            radius=self.circle_radius,
            num_obstacles=self.num_obstacles,
            min_size=min_obstacle_size,
            max_size=max_obstacle_size,
            offset_degrees=offset_degrees,
            passage_width=passage_width,
            level=self.obstacle_level
        )

        # Initialize swarm
        start_position = self.circle_center + self.circle_radius * np.array([1,0])
        if formation_type.lower() == 'triangle':
            self.positions = initialize_positions_triangle(num_robots, start_position, 5.0)
        else:
            self.positions = initialize_positions(num_robots, start_position, 5.0)
            # "5.0" is just a small initial radius for the arrangement

        self.headings = np.random.uniform(0, 2*np.pi, num_robots)
        self.velocities = np.zeros_like(self.positions)

        # track angles
        self.current_angle_accum = 0.0
        self.last_angle_accum = 0.0

        # figure out swarm center
        swarm_center = np.mean(self.positions, axis=0)
        rel = swarm_center - self.circle_center
        self.last_angle = np.arctan2(rel[1], rel[0])

        self.steps = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self._setup_episode()
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([
            self.positions.flatten(),
            self.headings.flatten(),
            self.velocities.flatten()
        ])

    def step(self, action):
        """
        RL sets [alpha, beta, K, C].
        Formation radius is fixed internally (self.formation_base).
        """
        alpha, beta, current_K, current_C = action

        # 1) The swarm center angle => path progress
        swarm_center = np.mean(self.positions, axis=0)
        rel_vec = swarm_center - self.circle_center
        angle_now = np.arctan2(rel_vec[1], rel_vec[0])

        # angle delta in [-pi, +pi], fix for boundary crossing
        delta_angle = angle_now - self.last_angle
        if delta_angle > np.pi:
            delta_angle -= 2*np.pi
        elif delta_angle < -np.pi:
            delta_angle += 2*np.pi

        self.current_angle_accum += delta_angle
        self.last_angle = angle_now

        # 2) Define the "moving center" based on self.current_angle_accum
        # The swarm tries to do 4 laps around the circle center
        moving_center = self.circle_center + self.circle_radius * np.array([
            np.cos(self.current_angle_accum),
            np.sin(self.current_angle_accum)
        ])

        # 3) target positions (using a fixed formation_base)
        if formation_type.lower() == 'triangle':
            target_positions = initialize_positions_triangle(num_robots, moving_center, self.formation_base)
        else:
            target_positions = initialize_positions(num_robots, moving_center, self.formation_base)

        # 4) forces
        forces, new_K, new_C = compute_forces_with_sensors(
            self.positions,
            self.headings,
            self.velocities,
            target_positions,
            self.obstacles,
            current_K, current_C,
            alpha, beta
        )

        # 5) update positions
        self.positions, self.headings = update_positions_and_headings(
            self.positions, self.headings, forces,
            self.robot_max_speed,
            (self.world_width, 0.5)
        )
        self.velocities = update_velocities(forces, self.robot_max_speed)

        # collisions
        _, collision_indices = check_collisions(np.copy(self.positions), self.obstacles)

        # --- Heavier collision penalty & survival bonus ---
        done = False
        truncated = False
        reward = 0.0
        
        # (Optional) penalty for cutting inside the circle:
        distances_to_center = np.linalg.norm(self.positions - self.circle_center, axis=1)
        # min_allowed_radius = self.circle_radius * 0.5
        # too_close_mask = distances_to_center < min_allowed_radius
        # penalty_for_cutting = 10.0
        # if np.any(too_close_mask):
        #     reward -= penalty_for_cutting * np.sum(too_close_mask)

        # Reward for staying on the path
        optimal_radius = self.circle_radius
        distance_error = np.abs(distances_to_center - optimal_radius)
        reward_for_path_following = np.exp(-distance_error)  
        reward += np.mean(reward_for_path_following) * 15.0  

        if not collision_indices:
            reward -= 200.0 * len(collision_indices)
            # done = True  # end if collision

        # 6) compute alignment/cost
        psi = compute_swarm_alignment(self.headings)
        sum_forces = np.sum(np.linalg.norm(forces, axis=1))
        time_progress = self.steps / float(num_steps * 2)
        cost_step = (
            cost_w1 * (1 - psi)**2
            + cost_w2 * time_progress
            + cost_w3 * sum_forces
        )
        reward -= cost_step
        reward += 5.0 * psi

        # Calculate a proximity penalty for each robot
        proximity_penalty = 0.0
        for pos in self.positions:
            for obs in self.obstacles:
                dist_to_obs = np.linalg.norm(pos - obs["position"]) - obs["radius"]
                safe_distance = sensor_detection_distance * 0.5  # for example
                if dist_to_obs < safe_distance:
                    # Penalize more as the robot gets closer to the obstacle
                    proximity_penalty += np.exp(-dist_to_obs)
        reward -= proximity_penalty * 200  # obstacle_penalty_weight to tune


        # path progress & completion bonus
        angle_progress = abs(self.current_angle_accum) - self.last_angle_accum
        if angle_progress > 0:
            reward += 1.0 * angle_progress
        self.last_angle_accum = abs(self.current_angle_accum)

        # if abs(self.current_angle_accum) >= self.max_angle:
        #     reward += 500.0
        #     done = True

        self.steps += 1
        if self.steps >= (num_steps * 4):
            truncated = True

        obs = self._get_obs()
        info = {}
        return obs, reward, done, truncated, info

    def render(self, mode='rgb_array'):
        return np.zeros((512,512,3), dtype=np.uint8)
