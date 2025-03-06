import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

# Import your environment class
from new_envV2 import SwarmEnv

# Import needed functions and constants from new_ocm
from new_ocmV2 import (
    num_steps, world_width, world_boundary_tolerance, robot_max_speed,
    num_robots, num_obstacles, min_obstacle_size, max_obstacle_size, sensor_detection_distance,
    offset_degrees, passage_width, obstacle_level,
    formation_radius_base, formation_size_triangle_base, formation_type,
    check_collisions, compute_forces_with_sensors, update_positions_and_headings,
    get_target_positions, get_target_positions_triangle, get_moving_center,
    generate_varied_obstacles_with_levels, adapt_parameters,
    compute_swarm_alignment, cost_w_align, cost_w_path, cost_w_obs, cost_w_force,
    alpha_base, beta_base, K_base, C_base,
    # Global arrays for plotting (optional)
    K_values_over_time, C_values_over_time,
    alpha_values_over_time, beta_values_over_time,
    total_cost,
    initialize_positions,
    initialize_positions_triangle,
    circle_center, circle_radius, obstacle_level,
)

###############################################################################
# Callback for Offline Video Generation Every N Episodes
###############################################################################
class OfflineVideoEveryNEpisodes(BaseCallback):
    """
    Callback for Stable Baselines3 that:
      1) Accumulates per-episode reward.
      2) Every video_episode_freq episodes, runs an offline simulation rollout 
         (using the current policy) and saves a video.
      3) Logs episode rewards.
    """
    def __init__(self, video_episode_freq=10, save_path="./videos", log_path="./episode_rewards_log.txt", verbose=1):
        super().__init__(verbose)
        self.video_episode_freq = video_episode_freq
        self.save_path = save_path
        self.log_path = log_path
        os.makedirs(save_path, exist_ok=True)
        self.episode_reward = 0.0
        self.episode_rewards = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]
        self.episode_reward += reward

        if done:
            self.episode_count += 1
            with open(self.log_path, "a") as f:
                f.write(f"Episode {self.episode_count}, Reward: {self.episode_reward}\n")
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0.0

            if (self.episode_count % self.video_episode_freq) == 0:
                last_ep_reward = self.episode_rewards[-1] if self.episode_rewards else 0.0
                video_filename = os.path.join(self.save_path, f"offline_sim_ep{self.episode_count}_r{last_ep_reward:.2f}.mp4")
                if self.verbose > 0:
                    print(f"[OfflineVideoEveryNEpisodes] Generating offline simulation for episode {self.episode_count}...")
                self.offline_playback(self.model, video_filename, self.episode_count, last_ep_reward)
        return True

    def offline_playback(self, model, filename, episode_idx, last_ep_reward):
        """
        Run an offline rollout using the current policy.
        Adaptive formation parameters are computed each step.
        """
        start_position = circle_center + circle_radius * np.array([1, 0])
        if formation_type.lower() == 'triangle':
            positions = initialize_positions_triangle(num_robots, start_position, formation_size_triangle_base)
        else:
            positions = initialize_positions(num_robots, start_position, formation_radius_base)
        headings = np.random.uniform(0, 2 * np.pi, num_robots)
        velocities = np.zeros_like(positions)
        obstacles = generate_varied_obstacles_with_levels(
            circle_center, circle_radius,
            num_obstacles, min_obstacle_size, max_obstacle_size,
            offset_degrees, passage_width, obstacle_level
        )

        fig, ax = plt.subplots()
        scatter = ax.scatter(positions[:, 0], positions[:, 1], c='blue', label='Robots')
        ax.add_artist(plt.Circle(circle_center, circle_radius, color='black', fill=False))
        for obs in obstacles:
            if obs["type"] == "circle":
                ax.add_artist(plt.Circle(obs["position"], obs["radius"], color='red', fill=True))
        ax.set_xlim(0, world_width)
        ax.set_ylim(0, world_width)
        ax.set_aspect('equal')
        ax.set_title(f"Offline Playback @ Episode {episode_idx} | LastEpReward={last_ep_reward:.2f}")
        count_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='darkred')

        max_frames = num_steps * 4
        local_step = 0
        # Initialize RL parameters to base values (can be adjusted if needed)
        alpha = cost_w_align  
        beta = cost_w_obs
        current_K = K_base
        current_C = C_base

        def animate(_frame):
            nonlocal positions, headings, velocities, local_step, alpha, beta, current_K, current_C

            # Create observation vector
            obs_vec = np.concatenate([positions.flatten(), headings.flatten(), velocities.flatten()])
            action, _ = model.predict(obs_vec, deterministic=True)
            alpha_rl, beta_rl, K_rl, C_rl = action
            alpha, beta, current_K, current_C = alpha_rl, beta_rl, K_rl, C_rl

            moving_center = get_moving_center(local_step, num_steps, positions)
            new_fr, new_fst, new_alpha, new_beta = adapt_parameters(
                positions, obstacles, formation_radius_base, formation_size_triangle_base, alpha, beta
            )
            if formation_type.lower() == 'triangle':
                t_positions = get_target_positions_triangle(moving_center, len(positions), new_fst)
            else:
                t_positions = get_target_positions(moving_center, len(positions), new_fr)

            forces, updated_K, updated_C = compute_forces_with_sensors(
                positions, headings, velocities,
                t_positions, obstacles,
                current_K, current_C, alpha, beta
            )
            positions, headings = update_positions_and_headings(
                positions, headings, forces,
                robot_max_speed, (world_width, world_boundary_tolerance)
            )
            speeds = np.linalg.norm(forces, axis=1)
            clipped_forces = [(robot_max_speed / s * fvec) if s > robot_max_speed else fvec
                              for s, fvec in zip(speeds, forces)]
            velocities[:] = np.array(clipped_forces)

            _, collisions = check_collisions(np.copy(positions), obstacles)
            scatter.set_offsets(positions)
            count_text.set_text(f"Robots remaining: {num_robots - len(collisions)}")

            local_step += 1
            if local_step >= max_frames or len(positions) == 0:
                plt.close(fig)
            return scatter,

        ani = animation.FuncAnimation(fig, animate, frames=max_frames, interval=50, blit=False, repeat=False)
        ani.save(filename, writer="ffmpeg")
        plt.close(fig)
        print(f"[OfflineVideoEveryNEpisodes] Video saved as {filename}")

###############################################################################
# Final Offline Simulation for Summary Plots
###############################################################################
def final_offline_simulation(model):
    """
    Runs a single offline simulation using the final policy.
    Adaptive formation parameters are computed each step.
    Summary plots for parameters and cost evolution are generated.
    """
    # Clear global arrays for plotting if needed
    K_values_over_time.clear()
    C_values_over_time.clear()
    alpha_values_over_time.clear()
    beta_values_over_time.clear()
    cost_history = []

    start_position = circle_center + circle_radius * np.array([1, 0])
    if formation_type.lower() == 'triangle':
        positions = initialize_positions_triangle(num_robots, start_position, formation_size_triangle_base)
    else:
        positions = initialize_positions(num_robots, start_position, formation_radius_base)
    headings = np.random.uniform(0, 2 * np.pi, num_robots)
    velocities = np.zeros_like(positions)
    obstacles = generate_varied_obstacles_with_levels(
        circle_center, circle_radius,
        num_obstacles, min_obstacle_size, max_obstacle_size,
        offset_degrees, passage_width, obstacle_level
    )

    fig, ax = plt.subplots()
    scatter = ax.scatter(positions[:, 0], positions[:, 1], c='blue')
    ax.add_artist(plt.Circle(circle_center, circle_radius, color='black', fill=False))
    for obs in obstacles:
        if obs["type"] == "circle":
            ax.add_artist(plt.Circle(obs["position"], obs["radius"], color='red', fill=True))
    ax.set_xlim(0, world_width)
    ax.set_ylim(0, world_width)
    ax.set_aspect('equal')
    count_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='darkred')

    alpha = alpha_base
    beta = beta_base
    current_K = K_base
    current_C = C_base

    def animate(frame):
        nonlocal positions, headings, velocities, alpha, beta, current_K, current_C
        global total_cost

        updated_positions, removed_indices = check_collisions(positions, obstacles)
        if removed_indices:
            headings = np.delete(headings, removed_indices, axis=0)
            velocities = np.delete(velocities, removed_indices, axis=0)
            positions = updated_positions

        if len(positions) == 0:
            scatter.set_offsets([])
            count_text.set_text("Robots remaining: 0")
            if frame == num_steps - 1:
                plt.close(fig)
            return scatter,

        obs_vec = np.concatenate([positions.flatten(), headings.flatten(), velocities.flatten()])
        action, _ = model.predict(obs_vec, deterministic=True)
        alpha_rl, beta_rl, K_rl, C_rl = action
        alpha, beta, current_K, current_C = alpha_rl, beta_rl, K_rl, C_rl

        moving_center = get_moving_center(frame, num_steps, positions)
        new_fr, new_fst, new_alpha, new_beta = adapt_parameters(
            positions, obstacles, formation_radius_base, formation_size_triangle_base, alpha, beta
        )
        if formation_type.lower() == 'triangle':
            t_positions = get_target_positions_triangle(moving_center, len(positions), new_fst)
        else:
            t_positions = get_target_positions(moving_center, len(positions), new_fr)

        forces, updK, updC = compute_forces_with_sensors(
            positions, headings, velocities,
            t_positions, obstacles,
            current_K, current_C, alpha, beta
        )
        current_K, current_C = updK, updC

        positions, headings = update_positions_and_headings(
            positions, headings, forces, robot_max_speed, (world_width, world_boundary_tolerance)
        )
        speeds = np.linalg.norm(forces, axis=1)
        clipped_forces = [(robot_max_speed / s * fvec) if s > robot_max_speed else fvec 
                          for s, fvec in zip(speeds, forces)]
        velocities[:] = np.array(clipped_forces)

        scatter.set_offsets(positions)
        count_text.set_text(f"Robots remaining: {len(positions)}")

        # Compute cost for logging
        psi = compute_swarm_alignment(headings)
        alignment_cost = cost_w_align * (1.0 - psi)**2
        path_errors = np.linalg.norm(positions - t_positions, axis=1)
        avg_path_error = np.mean(path_errors) if len(path_errors) > 0 else 0.0
        path_cost = cost_w_path * avg_path_error
        collision_cost = 0.0
        safe_distance = sensor_detection_distance / 2.0
        for i in range(len(positions)):
            for obs in obstacles:
                obs_dist = np.linalg.norm(positions[i] - obs["position"]) - obs["radius"]
                if obs_dist < safe_distance:
                    collision_cost += cost_w_obs * np.exp(-obs_dist)
        sum_force = np.sum(np.linalg.norm(forces, axis=1))
        control_cost = cost_w_force * sum_force
        cost_step = alignment_cost + path_cost + collision_cost + control_cost
        total_cost += cost_step
        cost_history.append(total_cost)

        K_values_over_time.append(current_K)
        C_values_over_time.append(current_C)
        alpha_values_over_time.append(alpha)
        beta_values_over_time.append(beta)
        return scatter,

    ani = animation.FuncAnimation(fig, animate, frames=num_steps, interval=50, repeat=False)
    plt.show()

    print(f"Robots left after simulation: {len(positions)}")
    print(f"Final accumulated cost: {total_cost:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(K_values_over_time, label='Alignment (K)', color='blue')
    plt.plot(C_values_over_time, label='Cohesion (C)', color='green')
    plt.xlabel('Time Step')
    plt.ylabel('Values')
    plt.title('Alignment (K) and Cohesion (C) Over Time')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values_over_time, label='Alpha', color='blue')
    plt.plot(beta_values_over_time, label='Beta', color='green')
    plt.xlabel('Time Step')
    plt.ylabel('Values')
    plt.title('Alpha and Beta Over Time')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, label='Accumulated Cost', color='red')
    plt.xlabel('Time Step')
    plt.ylabel('Accumulated Cost')
    plt.title('Cost Function Evolution Over Time')
    plt.legend()
    plt.show()

###############################################################################
# Main Training Logic
###############################################################################
def main():
    """
    Main training entry point:
      - Creates a single environment instance.
      - Trains a PPO model with the offline video callback.
      - Saves the trained model and generates a final offline simulation.
    """
    vec_env = make_vec_env(lambda: SwarmEnv(seed_value=42), n_envs=1)
    model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu")
    level = f"Level{obstacle_level}"  # Change this to the appropriate level as needed
    save_path = f"./videos/{level}"
    log_path = os.path.join(save_path, "episode_rewards_log.txt")
    video_callback = OfflineVideoEveryNEpisodes(video_episode_freq=10, save_path=save_path, log_path=log_path)

    # Clean existing logs and videos if desired
    if os.path.exists(video_callback.save_path):
        for file in os.listdir(video_callback.save_path):
            os.remove(os.path.join(video_callback.save_path, file))
    if os.path.exists(video_callback.log_path):
        os.remove(video_callback.log_path)

    print("Training the PPO model...")
    model.learn(total_timesteps=100000, callback=video_callback)
    model.save("swarm_navigation_policy")
    print("Training complete. Model saved as 'swarm_navigation_policy'.")

    last_ep = len(video_callback.episode_rewards)
    last_reward = video_callback.episode_rewards[-1] if last_ep > 0 else 0.0
    final_video = os.path.join("./videos", f"final_offline_run_ep{last_ep}.mp4")
    print(f"Generating final offline playback for ep {last_ep} ...")
    video_callback.offline_playback(model, final_video, last_ep, last_reward)

    with open(video_callback.log_path, "r") as f:
        lines = f.readlines()
        episode_rewards = [float(line.split()[-1]) for line in lines]

    window_size = 10
    avg_rewards = [np.mean(episode_rewards[i:i+window_size]) for i in range(0, len(episode_rewards), window_size)]
    plt.figure()
    plt.plot(avg_rewards, marker='o')
    plt.xlabel(f"Episode (grouped by {window_size})")
    plt.ylabel("Average Reward")
    plt.title("Average Episode Rewards Over Training")
    plt.grid()
    plt.savefig("average_episode_rewards_log.png")
    plt.show()

    print("\nRunning a final offline simulation for summary plots...\n")
    final_offline_simulation(model)

if __name__ == "__main__":
    main()
