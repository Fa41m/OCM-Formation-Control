import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

# Import your environment and simulation functions from your new modules.
from new_env import SwarmEnv
from new_ocm import (
    num_steps,
    check_collisions,
    formation_type,
    compute_forces_with_sensors,
    update_positions_and_headings,
    get_target_positions,
    get_target_positions_triangle,
    get_moving_center,
    # If your new_ocm exports base formation parameters under these names,
    # adjust as needed.
    formation_radius_base,
    width,
    num_robots,
    num_obstacles,
    min_obstacle_size,
    max_obstacle_size,
    offset_degrees,
    passage_width,
    obstacle_level,
    max_speed,
    world_boundary_tolerance,
    formation_size_triangle_base,
    # Default repulsion parameters (α and β)
    # (These might be renamed in your new_ocm; adjust if needed.)
    # For example, if you still call them alpha_base and beta_base:
    # alpha_base, beta_base,
)

# --- Offline Video Callback ---
class OfflineVideoCallback(BaseCallback):
    """
    Callback that:
    1) Accumulates per-episode reward,
    2) Periodically runs an offline simulation rollout (using the current policy)
       and saves a video with the last completed episode's reward in the filename/title,
    3) Logs each episode's reward to a text file.
    """
    def __init__(self, save_freq, save_path="./videos", log_path="./episode_rewards_log.txt", verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.log_path = log_path
        os.makedirs(save_path, exist_ok=True)

        self.episode_reward = 0.0
        self.episode_rewards = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Accumulate reward from the single (vectorized) environment.
        reward = self.locals["rewards"][0]  # (assumes one env)
        done = self.locals["dones"][0]

        self.episode_reward += reward

        # When an episode ends, log the reward.
        if done:
            self.episode_count += 1
            with open(self.log_path, "a") as f:
                f.write(f"Episode {self.episode_count}, Reward: {self.episode_reward}\n")
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0.0

        # Every save_freq calls, generate an offline simulation video.
        if self.n_calls % self.save_freq == 0:
            last_ep_reward = self.episode_rewards[-1] if self.episode_rewards else 0.0
            video_filename = os.path.join(
                self.save_path,
                f"offline_sim_{self.n_calls}_ep{self.episode_count}_r{last_ep_reward:.2f}.mp4"
            )
            print(f"[OfflineVideoCallback] Generating offline simulation at step {self.n_calls}...")
            self.offline_playback(self.model, video_filename, last_ep_reward)

        return True

    def offline_playback(self, model, filename, last_ep_reward):
        """
        Runs an offline rollout using the current RL policy (via model.predict),
        steps the simulation using your new_ocm functions, and saves a video.
        """
        # Import simulation functions needed for offline playback.
        # (Make sure these names match your new_ocm module.)
        from new_ocm import (
            initialize_positions,
            initialize_positions_triangle,
            circle_center,
            circle_radius,
            sensor_detection_distance,
            sensor_buffer_radius,
            generate_varied_obstacles_with_levels
        )

        # --- Setup for offline playback ---
        start_position = circle_center + circle_radius * np.array([1, 0])
        if formation_type == 'circle':
            positions = initialize_positions(num_robots, start_position, formation_radius_base)
        elif formation_type == 'triangle':
            positions = initialize_positions_triangle(num_robots, start_position, formation_size_triangle_base)
        headings = np.random.uniform(0, 2 * np.pi, num_robots)
        velocities = np.zeros_like(positions)

        obstacles = generate_varied_obstacles_with_levels(
            circle_center, circle_radius, num_obstacles,
            min_obstacle_size, max_obstacle_size, offset_degrees,
            passage_width, obstacle_level
        )

        fig, ax = plt.subplots()
        scatter = ax.scatter(positions[:, 0], positions[:, 1], c='blue', label='Robots')
        ax.add_artist(plt.Circle(circle_center, circle_radius, color='black', fill=False))
        for obs in obstacles:
            if obs["type"] == "circle":
                circle_patch = plt.Circle(obs["position"], obs["radius"], color='red', fill=True)
                ax.add_artist(circle_patch)

        ax.set_xlim(0, width)
        ax.set_ylim(0, width)
        ax.set_aspect('equal')
        ax.set_title(f"Offline Playback @ {self.n_calls} steps | LastEpReward={last_ep_reward:.2f}")
        count_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='darkred')

        # We'll simulate for (num_steps * 4) frames (matching the episode length)
        max_frames = num_steps * 4
        local_step = 0

        # Set initial repulsion parameters. (You can change these or compute them adaptively.)
        current_K, current_C = 1.0, 1.0  # defaults; these will be overridden by model actions
        # If you export default repulsion values from new_ocm (for example alpha_base, beta_base), import and use them:
        # from new_ocm import alpha_base, beta_base
        alpha = 0.4  # or use alpha_base if available
        beta = 0.4   # or use beta_base if available

        def animate(_frame):
            nonlocal positions, headings, velocities, local_step, current_K, current_C

            # 1. Build observation (flattened state)
            obs = np.concatenate([
                positions.flatten(),
                headings.flatten(),
                velocities.flatten()
            ])
            # 2. Get action from the policy (agent controls K and C)
            action, _states = model.predict(obs, deterministic=True)
            current_K, current_C = action

            # 3. Compute the moving center and target positions
            moving_center = get_moving_center(local_step, num_steps)
            if formation_type == 'circle':
                t_positions = get_target_positions(moving_center, len(positions), formation_radius_base)
            elif formation_type == 'triangle':
                t_positions = get_target_positions_triangle(moving_center, len(positions), formation_size_triangle_base)

            # 4. Compute forces using the new simulation function.
            # Note: Our new compute_forces_with_sensors returns (forces, current_K, current_C)
            forces, current_K, current_C = compute_forces_with_sensors(
                positions, headings, velocities, t_positions,
                obstacles, current_K, current_C, alpha, beta
            )

            # Optionally, check collisions (but do not remove robots here)
            _, collisions = check_collisions(np.copy(positions), obstacles)

            # 5. Update positions and headings.
            positions, headings = update_positions_and_headings(
                positions, headings, forces, max_speed, (width, world_boundary_tolerance)
            )
            # Update velocities (this example simply uses the clipped forces as velocities)
            velocities = np.array([ (max_speed/np.linalg.norm(f) * f) if np.linalg.norm(f) > max_speed else f for f in forces ])

            # 6. Update the scatter plot and text.
            scatter.set_offsets(positions)
            count_text.set_text(f"Robots remaining: {num_robots - len(collisions)}")

            local_step += 1
            if len(positions) == 0 or local_step >= max_frames:
                plt.close(fig)
            return scatter,

        ani = animation.FuncAnimation(
            fig, animate, frames=max_frames, interval=50, blit=False, repeat=False
        )
        ani.save(filename, writer="ffmpeg")
        plt.close(fig)
        print(f"Video saved as {filename}")

# --- Main training function ---
def main():
    # 1. Create the vectorized environment.
    vec_env = make_vec_env(lambda: SwarmEnv(seed_value=42, episode_length_factor=4), n_envs=1)

    # 2. Create the PPO model.
    model = PPO("MlpPolicy", vec_env, verbose=1)

    # 3. Create the callback for offline video saving.
    video_callback = OfflineVideoCallback(
        save_freq=1000,
        save_path="./videos",
        log_path="./episode_rewards_log.txt"
    )

    # Clean any existing video files and log.
    if os.path.exists(video_callback.save_path):
        for file in os.listdir(video_callback.save_path):
            os.remove(os.path.join(video_callback.save_path, file))
    if os.path.exists(video_callback.log_path):
        os.remove(video_callback.log_path)

    print("Training the PPO model...")
    model.learn(total_timesteps=40000, callback=video_callback)
    model.save("swarm_navigation_policy")
    print("Training complete. Model saved as 'swarm_navigation_policy'.")

    # 5. Optionally, generate a final offline playback video.
    final_video = os.path.join("./videos", "final_offline_run.mp4")
    print("Generating final offline playback...")
    last_reward = video_callback.episode_rewards[-1] if video_callback.episode_rewards else 0.0
    video_callback.offline_playback(model, final_video, last_reward)

    # 6. Plot the episode rewards log.
    with open(video_callback.log_path, "r") as f:
        lines = f.readlines()
        episode_rewards = [float(line.split()[-1]) for line in lines]
        avg_rewards = [np.mean(episode_rewards[i:i+10]) for i in range(0, len(episode_rewards), 10)]
        plt.plot(avg_rewards)
        plt.xlabel("Episode (in tens)")
        plt.ylabel("Average Reward")
        plt.title("Average Episode Rewards Log (per 10 episodes)")
        plt.grid()
        plt.savefig("average_episode_rewards_log.png")
        plt.show()

if __name__ == "__main__":
    main()
