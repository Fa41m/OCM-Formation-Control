import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from env import SwarmEnv
from ocm import (
    # The standard pieces from your original code
    num_steps,
    check_collisions,
    compute_forces_with_sensors,
    update_positions_and_headings,
    get_target_positions,
    get_moving_center,
    formation_radius,
    width,
    num_robots,
    num_obstacles,
    min_obstacle_size,
    max_obstacle_size,
    offset_degrees,
    passage_width,
    obstacle_level
)

# --------------------------------------------------------------------- #
#   1. Train the model with the minimal environment
# --------------------------------------------------------------------- #

class OfflineVideoCallback(BaseCallback):
    """
    Callback to periodically run an offline simulation
    where we do collision checks and create a video of
    the swarm's performance so far.
    """
    def __init__(self, save_freq, save_path="./videos", verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            video_filename = os.path.join(self.save_path, f"offline_sim_{self.n_calls}.mp4")
            print(f"[OfflineVideoCallback] Generating offline simulation at step {self.n_calls}...")
            self.offline_playback(self.model, video_filename)
        return True

    def offline_playback(self, model, filename):
        """
        Runs an offline rollout using the RL policy
        but does collision checks + removal,
        and saves a video of the swarm's performance.
        """
        # 1. We do a fresh "swarm" from the OCM code, re-initialized
        #    exactly how we want for playback
        from ocm import (
            initialize_positions, circle_center, circle_radius, 
            sensor_detection_distance, sensor_buffer_radius,
            generate_varied_obstacles_with_levels
        )

        # Setup for offline playback
        start_position = circle_center + circle_radius * np.array([1, 0])
        positions = initialize_positions(num_robots, start_position, formation_radius)
        headings = np.random.uniform(0, 2 * np.pi, num_robots)
        velocities = np.zeros_like(positions)
        obstacles = generate_varied_obstacles_with_levels(
            circle_center, circle_radius, num_obstacles, min_obstacle_size, max_obstacle_size, offset_degrees, passage_width, obstacle_level
        )

        fig, ax = plt.subplots()
        scatter = ax.scatter(positions[:, 0], positions[:, 1], c='blue', label='Robots')
        ax.add_artist(plt.Circle(circle_center, circle_radius, color='black', fill=False))

        # Draw obstacles
        for obs in obstacles:
            if obs["type"] == "circle":
                circle = plt.Circle(obs["position"], obs["radius"], color='red', fill=True)
                ax.add_artist(circle)

        ax.set_xlim(0, width)
        ax.set_ylim(0, width)
        ax.set_aspect('equal')
        ax.set_title(f"Offline Playback at Timestep {self.n_calls}")
        count_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='darkred')

        # We'll step up to num_steps frames
        max_frames = num_steps

        # We'll keep a local "env-like" step counter
        local_step = 0

        # The animation function
        def animate(_frame):
            nonlocal positions, headings, velocities, local_step
            # 1. RL agent chooses action from the environment observation
            #    We'll craft an observation that matches env._get_obs():
            obs = np.concatenate([
                positions.flatten(),
                headings.flatten(),
                velocities.flatten()
            ])
            action, _states = model.predict(obs, deterministic=True)

            # 2. Convert action -> K, C
            current_K, current_C = action

            # 3. Move center & compute target positions
            moving_center = get_moving_center(local_step, num_steps)
            target_positions = get_target_positions(
                moving_center, len(positions), formation_radius
            )

            # 4. Use OCM sensor code to get forces
            forces, _, _, _ = compute_forces_with_sensors(
                positions, headings, velocities, target_positions,
                obstacles, current_K, current_C
            )

            # 5. Update positions
            positions, headings = update_positions_and_headings(
                positions, headings, forces, 0.3, (width, 0.5)
            )

            # 6. Check collisions & remove collided robots
            # positions, num_removed = check_collisions(positions, obstacles)

            # 7. Update scatter + count
            scatter.set_offsets(positions)
            count_text.set_text(f"Robots remaining: {len(positions)}")

            local_step += 1

            # If all robots are gone or we reached max frames, we end
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
    # Create a vectorized environment (1 env)
    vec_env = make_vec_env(lambda: SwarmEnv(seed_value=42), n_envs=1)

    model = PPO("MlpPolicy", vec_env, verbose=1)

    # Create a callback that does offline playback every X steps
    video_callback = OfflineVideoCallback(
        save_freq=1000,  # e.g. every 15k training steps
        save_path="./videos"
    )

    print("Training the PPO model...")
    model.learn(total_timesteps=40000, callback=video_callback)
    model.save("swarm_navigation_policy")
    print("Training complete. Model saved as 'swarm_navigation_policy'.")

    # Now, do one final offline playback after training
    final_video = os.path.join("./videos", "final_offline_run.mp4")
    print("Generating final offline playback...")
    video_callback.offline_playback(model, final_video)

if __name__ == "__main__":
    main()
