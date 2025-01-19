import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from env import SwarmEnv
from ocm import (
    num_steps,
    check_collisions,
    formation_type,
    compute_forces_with_sensors,
    update_positions_and_headings,
    get_target_positions,
    get_target_positions_triangle,
    get_moving_center,
    formation_radius,
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
    formation_size_triangle,
)

class OfflineVideoCallback(BaseCallback):
    """
    Callback to:
    1) Accumulate per-episode reward
    2) Periodically run an offline simulation and save the video
       while embedding the most recently completed episode's reward in
       the video title (and/or filename).
    3) Log each episode's reward to a text file.
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
        """
        Called at every training step by Stable Baselines.
        We accumulate the reward, detect episode boundaries,
        log to a file, and possibly trigger offline video generation.
        """
        # 1. Accumulate reward from the single (vectorized) env
        reward = self.locals["rewards"][0]  # 1 env
        done = self.locals["dones"][0]      # 1 env

        self.episode_reward += reward

        # 2. If the episode just ended, log the reward and reset
        if done:
            self.episode_count += 1

            # Write to log file
            with open(self.log_path, "a") as f:
                f.write(f"Episode {self.episode_count}, Reward: {self.episode_reward}\n")

            self.episode_rewards.append(self.episode_reward)
            # Reset for next episode
            self.episode_reward = 0.0

        # 3. Check if it's time to save a video
        if self.n_calls % self.save_freq == 0:
            # We'll embed the last completed episode's reward in the filename/title
            last_ep_reward = self.episode_rewards[-1] if len(self.episode_rewards) > 0 else 0.0

            # Example: put it in the filename
            video_filename = os.path.join(
                self.save_path,
                f"offline_sim_{self.n_calls}_ep{self.episode_count}_r{last_ep_reward:.2f}.mp4"
            )

            print(f"[OfflineVideoCallback] Generating offline simulation at step {self.n_calls}...")
            self.offline_playback(self.model, video_filename, last_ep_reward)

        return True

    def offline_playback(self, model, filename, last_ep_reward):
        """
        Runs an offline rollout using the RL policy (model.predict),
        does collision checks, and saves a video of the swarm's performance.
        We'll also put the reward in the video title.
        """
        from ocm import (
            initialize_positions, initialize_positions_triangle, circle_center, circle_radius,
            sensor_detection_distance, sensor_buffer_radius,
            generate_varied_obstacles_with_levels
        )

        # Setup for offline playback
        start_position = circle_center + circle_radius * np.array([1, 0])
        if formation_type == 'circle':
            positions = initialize_positions(num_robots, start_position, formation_radius)
        elif formation_type == 'triangle':
            positions = initialize_positions_triangle(num_robots, start_position, formation_size_triangle)
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
                circle = plt.Circle(obs["position"], obs["radius"], color='red', fill=True)
                ax.add_artist(circle)

        ax.set_xlim(0, width)
        ax.set_ylim(0, width)
        ax.set_aspect('equal')
        # Show the last completed episode's reward in the title
        ax.set_title(f"Offline Playback @ {self.n_calls} steps | LastEpReward={last_ep_reward:.2f}")
        count_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='darkred')

        # We'll step up to num_steps * 4 frames (the same as your new episode length)
        max_frames = num_steps * 4

        local_step = 0

        def animate(_frame):
            nonlocal positions, headings, velocities, local_step
            # 1. Build observation
            obs = np.concatenate([
                positions.flatten(),
                headings.flatten(),
                velocities.flatten()
            ])
            # 2. Get action from policy
            action, _states = model.predict(obs, deterministic=True)
            current_K, current_C = action

            # 3. Move center & compute target positions
            moving_center = get_moving_center(local_step, num_steps)
            if formation_type == 'circle':
                t_positions = get_target_positions(moving_center, len(positions), formation_radius)
            elif formation_type == 'triangle':
                t_positions = get_target_positions_triangle(moving_center, len(positions), formation_size_triangle)

            # 4. Compute forces (with collisions)
            forces, _, _, _, collisions = compute_forces_with_sensors(
                positions, headings, velocities, t_positions,
                obstacles, current_K, current_C
            )

            # 5. Update positions
            positions, headings = update_positions_and_headings(
                positions, headings, forces, max_speed, (width, world_boundary_tolerance)
            )

            # 6. Update scatter
            scatter.set_offsets(positions)
            count_text.set_text(f"Robots remaining: {num_robots - len(collisions)}")
            
            # 7. Remove collided robots
            # positions = np.delete(positions, collisions, axis=0)
            # headings = np.delete(headings, collisions)
            # velocities = np.delete(velocities, collisions, axis=0)

            local_step += 1
            # Terminate if we exceed max_frames or lose all robots
            if len(positions) == 0 or local_step >= max_frames:
                plt.close(fig)
            return scatter,

        ani = animation.FuncAnimation(
            fig, animate, frames=max_frames, interval=50, blit=False, repeat=False
        )
        ani.save(filename, writer="ffmpeg")
        plt.close(fig)
        print(f"Video saved as {filename}")


def main():
    # 1. Create environment with each episode = num_steps*4
    vec_env = make_vec_env(lambda: SwarmEnv(seed_value=42, episode_length_factor=4), n_envs=1)

    # 2. Create the PPO model
    model = PPO("MlpPolicy", vec_env, verbose=1)

    # 3. Create the callback with video saving frequency = 10000 timesteps
    video_callback = OfflineVideoCallback(
        save_freq=5000,
        save_path="./videos",
        log_path="./episode_rewards_log.txt"
    )

    print("Training the PPO model...")
    # 4. Train the model for some total timesteps
    model.learn(total_timesteps=40000, callback=video_callback)
    model.save("swarm_navigation_policy")
    print("Training complete. Model saved as 'swarm_navigation_policy'.")

    # 5. Optionally do a final offline playback
    final_video = os.path.join("./videos", "final_offline_run.mp4")
    print("Generating final offline playback...")
    # Just pass in the last known reward, or 0 if none
    last_reward = video_callback.episode_rewards[-1] if len(video_callback.episode_rewards) > 0 else 0.0
    video_callback.offline_playback(model, final_video, last_reward)
    
    # plot the episode rewards log
    with open(video_callback.log_path, "r") as f:
        lines = f.readlines()
        episode_rewards = [float(line.split()[-1]) for line in lines]
        plt.plot(episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Episode Rewards Log")
        plt.grid()
        plt.savefig("episode_rewards_log.png")
        plt.show()

if __name__ == "__main__":
    main()
