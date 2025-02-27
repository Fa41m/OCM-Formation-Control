import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

# Import your environment class
from new_env import SwarmEnv

# Import from new_ocm whatever is needed for offline rollout + final summary
from new_ocm import (
    num_steps, world_width, world_boundary_tolerance, robot_max_speed,
    num_robots, num_obstacles, min_obstacle_size, max_obstacle_size,
    offset_degrees, passage_width, obstacle_level,
    formation_radius_base, formation_size_triangle_base, formation_type,
    check_collisions, compute_forces_with_sensors, update_positions_and_headings,
    get_target_positions, get_target_positions_triangle, get_moving_center,
    generate_varied_obstacles_with_levels,
    # For the final summary, we need cost references & arrays
    compute_swarm_alignment, cost_w1, cost_w2, cost_w3, psi_threshold,
    K_values_over_time, C_values_over_time,
    alpha_values_over_time, beta_values_over_time,
    t_rise, t_rise_recorded, total_cost
)

###############################################################################
# 1) OfflineVideoEveryNEpisodes Callback: video every N episodes
###############################################################################
class OfflineVideoEveryNEpisodes(BaseCallback):
    """
    A callback for Stable Baselines3 that:
     1) Accumulates per-episode reward.
     2) Every `video_episode_freq` episodes, runs an offline simulation rollout 
        using the *current policy* to produce a .mp4 video.
     3) Logs the episode rewards in a text file.
    """

    def __init__(
        self,
        video_episode_freq=10,
        save_path="./videos",
        log_path="./episode_rewards_log.txt",
        verbose=1
    ):
        super().__init__(verbose)
        self.video_episode_freq = video_episode_freq
        self.save_path = save_path
        self.log_path = log_path
        os.makedirs(save_path, exist_ok=True)

        self.episode_reward = 0.0
        self.episode_rewards = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # We assume a single environment (n_envs=1)
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]

        self.episode_reward += reward

        if done:
            # End of an episode
            self.episode_count += 1

            # Log the total episode reward
            with open(self.log_path, "a") as f:
                f.write(f"Episode {self.episode_count}, Reward: {self.episode_reward}\n")

            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0.0

            # Every N episodes => offline rollout video
            if (self.episode_count % self.video_episode_freq) == 0:
                last_ep_reward = self.episode_rewards[-1] if self.episode_rewards else 0.0
                video_filename = os.path.join(
                    self.save_path,
                    f"offline_sim_ep{self.episode_count}_r{last_ep_reward:.2f}.mp4"
                )
                if self.verbose > 0:
                    print(f"[OfflineVideoEveryNEpisodes] Generating offline simulation for episode {self.episode_count}...")
                self.offline_playback(self.model, video_filename, self.episode_count, last_ep_reward)

        return True

    def offline_playback(self, model, filename, episode_idx, last_ep_reward):
        """
        Runs an offline rollout using the current RL policy (model.predict),
        stepping a *manual* simulation with new_ocm’s methods.
        Saves a video with Matplotlib.
        """
        from new_ocm import (
            initialize_positions,
            initialize_positions_triangle,
            circle_center, circle_radius
        )

        start_position = circle_center + circle_radius * np.array([1, 0])
        if formation_type.lower() == 'triangle':
            positions = initialize_positions_triangle(
                num_robots, start_position, formation_size_triangle_base
            )
        else:  # "circle"
            positions = initialize_positions(
                num_robots, start_position, formation_radius_base
            )
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
                circle_patch = plt.Circle(obs["position"], obs["radius"], color='red', fill=True)
                ax.add_artist(circle_patch)

        ax.set_xlim(0, world_width)
        ax.set_ylim(0, world_width)
        ax.set_aspect('equal')
        ax.set_title(
            f"Offline Playback @ Episode {episode_idx} | LastEpReward={last_ep_reward:.2f}"
        )
        count_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='darkred')

        # We'll simulate for (num_steps * 4) frames, or until no robots remain
        max_frames = num_steps * 4
        local_step = 0

        def animate(_frame):
            nonlocal positions, headings, velocities, local_step

            obs = np.concatenate([
                positions.flatten(),
                headings.flatten(),
                velocities.flatten()
            ])

            # RL policy => (alpha, beta, K, C)
            action, _states = model.predict(obs, deterministic=True)
            alpha, beta, current_K, current_C = action

            # Use get_moving_center or your new 4-lap method 
            # (But for a simple offline playback, we might just do the old approach)
            moving_center = get_moving_center(local_step, num_steps, positions)

            if formation_type.lower() == 'triangle':
                t_positions = get_target_positions_triangle(
                    moving_center, len(positions), formation_size_triangle_base
                )
            else:
                t_positions = get_target_positions(
                    moving_center, len(positions), formation_radius_base
                )

            from new_ocm import compute_forces_with_sensors, update_positions_and_headings, check_collisions

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
            clipped_forces = []
            for i, fvec in enumerate(forces):
                if speeds[i] > robot_max_speed:
                    clipped_forces.append((robot_max_speed / speeds[i]) * fvec)
                else:
                    clipped_forces.append(fvec)
            velocities[:] = np.array(clipped_forces)

            _, collisions = check_collisions(np.copy(positions), obstacles)
            scatter.set_offsets(positions)
            count_text.set_text(
                f"Robots remaining: {num_robots - len(collisions)}"
            )

            local_step += 1
            if local_step >= max_frames or len(positions) == 0:
                plt.close(fig)
            return scatter,

        ani = animation.FuncAnimation(
            fig, animate, frames=max_frames,
            interval=50, blit=False, repeat=False
        )
        ani.save(filename, writer="ffmpeg")
        plt.close(fig)
        print(f"[OfflineVideoEveryNEpisodes] Video saved as {filename}")

###############################################################################
# 2) Final Offline Simulation that replicates new_ocm summary
###############################################################################
def final_offline_simulation(model):
    """
    Same as before: runs a single offline simulation using the final model,
    collects data (K,C,alpha,beta,cost) each step, replicates new_ocm plots.
    """
    # Clear out global arrays so we don't mix old runs
    K_values_over_time.clear()
    C_values_over_time.clear()
    alpha_values_over_time.clear()
    beta_values_over_time.clear()

    global t_rise, t_rise_recorded, total_cost
    t_rise = num_steps
    t_rise_recorded = False
    total_cost = 0.0
    cost_history = []

    from new_ocm import (
        initialize_positions,
        initialize_positions_triangle,
        circle_center, circle_radius,
    )

    # Initialize
    start_position = circle_center + circle_radius * np.array([1, 0])
    if formation_type.lower() == 'triangle':
        positions = initialize_positions_triangle(
            num_robots, start_position, formation_size_triangle_base
        )
    else:
        positions = initialize_positions(
            num_robots, start_position, formation_radius_base
        )
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
            circle_patch = plt.Circle(obs["position"], obs["radius"], color='red', fill=True)
            ax.add_artist(circle_patch)
    ax.set_xlim(0, world_width)
    ax.set_ylim(0, world_width)
    ax.set_aspect('equal')
    count_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='darkred')

    alpha = 0.4
    beta = 0.4
    current_K = 0.8
    current_C = 0.7

    def animate(frame):
        nonlocal positions, headings, velocities, alpha, beta, current_K, current_C
        global t_rise, t_rise_recorded, total_cost

        from new_ocm import check_collisions, compute_forces_with_sensors, update_positions_and_headings
        updated_positions, removed_indices = check_collisions(positions, obstacles)
        if removed_indices:
            headings_ = np.delete(headings, removed_indices, axis=0)
            velocities_ = np.delete(velocities, removed_indices, axis=0)
            positions_ = updated_positions
            headings[:] = headings_
            velocities[:] = velocities_
            positions[:] = positions_

        if len(positions) == 0:
            scatter.set_offsets([])
            count_text.set_text("Robots remaining: 0")
            if frame == num_steps - 1:
                plt.close(fig)
            return scatter,

        # RL action
        obs = np.concatenate([positions.flatten(), headings.flatten(), velocities.flatten()])
        action, _ = model.predict(obs, deterministic=True)
        alpha_rl, beta_rl, K_rl, C_rl = action
        alpha = alpha_rl
        beta = beta_rl
        current_K = K_rl
        current_C = C_rl

        # target positions
        moving_center = get_moving_center(frame, num_steps, positions)
        if formation_type.lower() == 'triangle':
            t_positions = get_target_positions_triangle(moving_center, len(positions), formation_size_triangle_base)
        else:
            t_positions = get_target_positions(moving_center, len(positions), formation_radius_base)

        forces, updK, updC = compute_forces_with_sensors(
            positions, headings, velocities,
            t_positions, obstacles,
            current_K, current_C,
            alpha, beta
        )
        current_K, current_C = updK, updC

        positions, headings = update_positions_and_headings(
            positions, headings, forces, robot_max_speed,
            (world_width, world_boundary_tolerance)
        )

        speeds = np.linalg.norm(forces, axis=1)
        clipped_forces = []
        for i, fvec in enumerate(forces):
            if speeds[i] > robot_max_speed:
                clipped_forces.append((robot_max_speed / speeds[i]) * fvec)
            else:
                clipped_forces.append(fvec)
        velocities[:] = np.array(clipped_forces)

        scatter.set_offsets(positions)
        count_text.set_text(f"Robots remaining: {len(positions)}")

        psi = compute_swarm_alignment(headings)
        if (not t_rise_recorded) and (psi >= psi_threshold):
            t_rise = frame
            t_rise_recorded = True

        sum_force = np.sum(np.linalg.norm(forces, axis=1))
        current_t_rise = t_rise if t_rise_recorded else num_steps
        cost_step = cost_w1*(1 - psi)**2 + cost_w2*(current_t_rise**2) + cost_w3*sum_force
        total_cost += cost_step
        cost_history.append(total_cost)

        # record K,C,alpha,beta
        K_values_over_time.append(current_K)
        C_values_over_time.append(current_C)
        alpha_values_over_time.append(alpha)
        beta_values_over_time.append(beta)

        return scatter,

    ani = animation.FuncAnimation(fig, animate, frames=num_steps, interval=50, repeat=False)
    plt.show()

    # Summary
    print(f"Robots left after simulation: {len(positions)}")
    print(f"Final accumulated cost: {total_cost:.2f}")
    print(f"Rise time (first frame where psi >= {psi_threshold}): {t_rise}")

    # K and C
    plt.figure(figsize=(10, 6))
    plt.plot(K_values_over_time, label='Alignment (K)', color='blue')
    plt.plot(C_values_over_time, label='Cohesion (C)', color='green')
    plt.xlabel('Time Step')
    plt.ylabel('Values')
    plt.title('Alignment (K) and Cohesion (C) Over Time')
    plt.legend()
    plt.show()

    # alpha and beta
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values_over_time, label='Alpha', color='blue')
    plt.plot(beta_values_over_time, label='Beta', color='green')
    plt.xlabel('Time Step')
    plt.ylabel('Values')
    plt.title('Alpha and Beta Over Time')
    plt.legend()
    plt.show()

    # cost evolution
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, label='Accumulated Cost', color='red')
    plt.xlabel('Time Step')
    plt.ylabel('Accumulated Cost')
    plt.title('Cost Function Evolution Over Time')
    plt.legend()
    plt.show()

###############################################################################
# 3) Main training logic
###############################################################################
def main():
    """
    Main training entry point:
      - The environment episodes end after 4 laps, collisions, or internal step limits
        (based on your new_env.py code).
      - We produce a video every 10 episodes.
    """

    # 1) Single env (remove episode_length_factor since SwarmEnv no longer expects it)
    vec_env = make_vec_env(
        lambda: SwarmEnv(seed_value=42),
        n_envs=1
    )

    # 2) Create PPO
    model = PPO("MlpPolicy", vec_env, verbose=1)

    # 3) Callback for offline video saving, triggered every 10 episodes
    video_callback = OfflineVideoEveryNEpisodes(
        video_episode_freq=10,
        save_path="./videos",
        log_path="./episode_rewards_log.txt"
    )

    # Clean existing logs if desired ...
    if os.path.exists(video_callback.save_path):
        for file in os.listdir(video_callback.save_path):
            os.remove(os.path.join(video_callback.save_path, file))
    if os.path.exists(video_callback.log_path):
        os.remove(video_callback.log_path)

    print("Training the PPO model...")

    # 4) The agent trains until some number of steps
    #    (But each episode length is determined by the environment logic, not this factor.)
    model.learn(total_timesteps=80000, callback=video_callback)
    model.save("swarm_navigation_policy")
    print("Training complete. Model saved as 'swarm_navigation_policy'.")

    # 5) Optionally do a final offline playback
    last_ep = len(video_callback.episode_rewards)
    last_reward = video_callback.episode_rewards[-1] if last_ep > 0 else 0.0
    final_video = os.path.join("./videos", f"final_offline_run_ep{last_ep}.mp4")
    print(f"Generating final offline playback for ep {last_ep} ...")
    video_callback.offline_playback(model, final_video, last_ep, last_reward)

    # 6) Plot the episode rewards log
    with open(video_callback.log_path, "r") as f:
        lines = f.readlines()
        episode_rewards = []
        for line in lines:
            parts = line.split()
            ep_reward = float(parts[-1])
            episode_rewards.append(ep_reward)

    window_size = 10
    avg_rewards = [
        np.mean(episode_rewards[i:i+window_size])
        for i in range(0, len(episode_rewards), window_size)
    ]

    plt.figure()
    plt.plot(avg_rewards, marker='o')
    plt.xlabel(f"Episode (grouped by {window_size})")
    plt.ylabel("Average Reward")
    plt.title("Average Episode Rewards Over Training")
    plt.grid()
    plt.savefig("average_episode_rewards_log.png")
    plt.show()

    # 7) final offline simulation => new_ocm summary plots
    print("\nRunning a final offline simulation for summary plots...\n")
    final_offline_simulation(model)

if __name__ == "__main__":
    main()
