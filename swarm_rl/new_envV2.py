import gymnasium as gym
import numpy as np

from new_ocmV2 import (
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
        #    (Here k_cohesion is a new parameter that you can tune.)
        k_cohesion = 0.01  # try adjusting this value (e.g. 0.05 to 0.2)
        cohesion_cost = k_cohesion * np.mean(np.linalg.norm(self.positions - swarm_center, axis=1)**2)
        reward -= cohesion_cost

        # 3. Control effort cost: as before.
        sum_force = np.sum(np.linalg.norm(forces, axis=1))
        control_cost = cost_w_force * sum_force
        reward -= control_cost

        if collision_indices:
            reward -= 300.0 * len(collision_indices)  # adjust if necessary
            
            
        distances_to_center = np.linalg.norm(self.positions - self.circle_center, axis=1)
        min_allowed_radius = self.circle_radius * 0.5
        too_close_mask = distances_to_center < min_allowed_radius
        penalty_for_cutting = 10.0
        if np.any(too_close_mask):
            reward -= penalty_for_cutting * np.sum(too_close_mask)


        # Reward for staying on the path:
        optimal_radius = self.circle_radius
        distance_error = np.abs(distances_to_center - optimal_radius)
        reward_for_path_following = np.exp(-distance_error)  
        reward += np.mean(reward_for_path_following)  


        # Calculate a proximity penalty for each robot (penalize closeness to obstacles):
        proximity_penalty = 0.0
        any_near_obstacle = False
        for pos in self.positions:
            for obs in self.obstacles:
                dist_to_obs = np.linalg.norm(pos - obs["position"]) - obs["radius"]
                if dist_to_obs < 2.0:  # or sensor_detection_distance * 0.5
                    any_near_obstacle = True
                if dist_to_obs < 1.0:
                    # the closer we get, the bigger the penalty
                    proximity_penalty += np.exp(-dist_to_obs)

        reward -= 30.0 * proximity_penalty

        # 3) Add a small formation bonus only if not near obstacles
        if not any_near_obstacle:
            # Suppose desired inter-robot distance is 5.0
            f_err = formation_error(self.positions, 5.0)
            f_good = np.exp(-f_err)  # 1.0 if perfect, decays with error
            reward += 0.5 * f_good  # small bonus, overshadowed by collision penalty if any
        
        distances = np.linalg.norm(self.positions - self.nominal_positions, axis=1)

        # 3) If a robot is “close enough” to its target, count it:
        # formation_tolerance = 2.0  # tune this
        # in_formation_mask = distances < formation_tolerance
        # in_formation_count = np.sum(in_formation_mask)

        # 4) Check if at least 60% of the swarm is in formation:
        # fraction_in_formation = in_formation_count / num_robots
        # if fraction_in_formation >= 0.60:
        #     # You can pick any reward bonus you like—here we add +10 as an example
        #     reward += 10.0


        # Path progress & completion bonus:
        angle_progress = abs(self.current_angle_accum) - self.last_angle_accum
        if angle_progress > 0:
            reward += 1.0 * angle_progress
        self.last_angle_accum = abs(self.current_angle_accum)

        if abs(self.current_angle_accum) >= self.max_angle:
            reward += 500.0
            done = True

        # Termination conditions (as before).
        done = False
        truncated = False
        if abs(self.current_angle_accum) >= self.max_angle:
            done = True
        self.steps += 1
        if self.steps >= (num_steps * 4):
            truncated = True

        obs = self._get_obs()
        info = {}
        return obs, reward, done, truncated, info


    def render(self, mode='rgb_array'):
        return np.zeros((512,512,3), dtype=np.uint8)
