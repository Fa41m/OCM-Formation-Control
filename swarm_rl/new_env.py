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
    cost_w_align, cost_w_path, cost_w_obs, cost_w_force,
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

def formation_error(positions, desired_distance):
    # positions: shape (N, 2)
    # desired_distance: float, e.g. 5.0
    N = len(positions)
    total_error = 0.0
    count = 0
    
    for i in range(N):
        for j in range(i+1, N):
            dist = np.linalg.norm(positions[i] - positions[j])
            total_error += abs(dist - desired_distance)
            count += 1
    
    avg_error = total_error / (count if count > 0 else 1)
    return avg_error

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
        self.nominal_positions = np.copy(self.positions)

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
        RL sets [alpha, beta, K, C] – but now these mainly affect how the agents respond to obstacles.
        The agents’ target is the current swarm centroid (to keep them together).
        """
        alpha, beta, current_K, current_C = action

        # Update swarm center (for other potential use, though our target here is simply cohesion)
        swarm_center = np.mean(self.positions, axis=0)
        
        # Set each agent’s target to the swarm centroid to encourage cohesion.
        target_positions = np.tile(swarm_center, (num_robots, 1))
        
        # Compute forces using your existing sensor-based force computation.
        forces, new_K, new_C = compute_forces_with_sensors(
            self.positions,
            self.headings,
            self.velocities,
            target_positions,
            self.obstacles,
            current_K, current_C,
            alpha, beta
        )
        current_K = new_K
        current_C = new_C

        # Update positions, headings, and velocities.
        self.positions, self.headings = update_positions_and_headings(
            self.positions, self.headings, forces,
            self.robot_max_speed,
            (self.world_width, 0.5)
        )
        self.velocities = update_velocities(forces, self.robot_max_speed)

        # Check for collisions (which will be used for a crash penalty).
        _, collision_indices = check_collisions(np.copy(self.positions), self.obstacles)

        # ---------------------------
        # Reward Shaping
        # ---------------------------
        reward = 0.0

        # 1. Cohesion cost: penalize agents that are far from the swarm center.
        k_cohesion = 0.01  # Tune this parameter as needed
        cohesion_cost = k_cohesion * np.mean(np.linalg.norm(self.positions - swarm_center, axis=1)**2)
        reward -= cohesion_cost

        # 2. Control effort cost: discourage excessive force usage.
        sum_force = np.sum(np.linalg.norm(forces, axis=1))
        control_cost = cost_w_force * sum_force
        reward -= control_cost

        # 3. Collision penalty: heavy penalty for every collision.
        if collision_indices:
            reward -= 300.0 * len(collision_indices)

        # 4. Path-following reward: encourage robots to stay close to the desired circular path.
        #    Normalize the distance error by the circle radius.
        distances_to_center = np.linalg.norm(self.positions - self.circle_center, axis=1)
        optimal_radius = self.circle_radius
        normalized_distance_error = np.abs(distances_to_center - optimal_radius) / optimal_radius
        # Reward gets close to 1 if error is near zero.
        reward_for_path_following = np.exp(-normalized_distance_error)
        reward += np.mean(reward_for_path_following)

        # 5. Penalty for cutting corners: if agents get too close to the center (i.e. "cutting the path").
        min_allowed_radius = 0.5 * self.circle_radius
        too_close_mask = distances_to_center < min_allowed_radius
        penalty_for_cutting = 10.0
        if np.any(too_close_mask):
            # Normalize by the number of robots so the penalty scales with swarm size.
            reward -= penalty_for_cutting * (np.sum(too_close_mask) / len(self.positions))

        # 6. Obstacle avoidance penalty: continuously penalize closeness to obstacles.
        obstacle_penalty_weight = 30.0
        proximity_penalty = 0.0
        for pos in self.positions:
            for obs in self.obstacles:
                # Calculate clearance from obstacle surface.
                dist_to_obs = np.linalg.norm(pos - obs["position"]) - obs["radius"]
                # Apply a smooth exponential penalty if within a threshold distance.
                threshold = 2.0  # Adjust this threshold if needed.
                if dist_to_obs < threshold:
                    proximity_penalty += np.exp(-dist_to_obs)
        # Average the penalty over the number of robots.
        reward -= obstacle_penalty_weight * (proximity_penalty / len(self.positions))

        # 7. Formation bonus: reward maintaining the desired formation when not in close proximity to obstacles.
        #    The formation error is computed externally (for example, average deviation from a desired inter-robot distance).
        #    This term is added only if agents are not near any obstacles.
        if proximity_penalty < 1e-3:  # You may adjust the threshold to define "safe" from obstacles.
            desired_inter_robot_distance = 5.0  # Tune as necessary.
            f_err = formation_error(self.positions, desired_inter_robot_distance)
            formation_bonus = np.exp(-f_err)  # Closer to 1 when formation is ideal.
            formation_bonus_weight = 0.5
            reward += formation_bonus_weight * formation_bonus

        # 8. Progress reward: reward for making progress along the circular path.
        angle_progress = abs(self.current_angle_accum) - self.last_angle_accum
        if angle_progress > 0:
            progress_reward_weight = 1.0
            reward += progress_reward_weight * angle_progress
        self.last_angle_accum = abs(self.current_angle_accum)
        
        done = False
        # 9. Termination bonus: provide a bonus upon full path completion.
        if abs(self.current_angle_accum) >= self.max_angle:
            # reward += 500.0
            done = True

        # # Termination conditions (as before).
        # done = False
        truncated = False
        self.steps += 1
        if self.steps >= (num_steps * 4):
            truncated = True

        obs = self._get_obs()
        info = {}
        return obs, reward, done, truncated, info


    def render(self, mode='rgb_array'):
        return np.zeros((512,512,3), dtype=np.uint8)
