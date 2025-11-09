# trajectory_runner.py
import numpy as np
import matplotlib.pyplot as plt
import ocm  # your module containing the sim primitives & constants

# === set this to your trained policy zip ===
POLICY_ZIP_PATH = "swarm_navigation_policy.zip"  # or None to disable PPO control

# (optional) Only needed if using PPO
try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None


def _build_obs(positions: np.ndarray, headings: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """
    Observation layout used during training:
      per-robot [x, y, heading, vx, vy], flattened to length 5*N.
    """
    return np.hstack([positions, headings[:, None], velocities]).astype(np.float32).ravel()


def simulate_and_record_trajectory_cfg(
    formation_type: str,
    obstacle_level: int,
    *,
    num_robots: int = 15,
    num_steps: int = 400,
    world_width_: float = None,
    robot_max_speed_: float = None,
    world_boundary_tolerance_: float = None,
    policy_zip: str | None = POLICY_ZIP_PATH,
):
    ww = ocm.world_width if world_width_ is None else world_width_
    rms = ocm.robot_max_speed if robot_max_speed_ is None else robot_max_speed_
    wbt = ocm.world_boundary_tolerance if world_boundary_tolerance_ is None else world_boundary_tolerance_

    # local moving-center state
    local_moving_theta = 0.0
    local_theta_step = 2.0 * np.pi / float(num_steps)

    def local_get_moving_center(swarm_positions: np.ndarray) -> np.ndarray:
        nonlocal local_moving_theta
        swarm_center = np.mean(swarm_positions, axis=0)
        expected_position = ocm.circle_center + ocm.circle_radius * np.array(
            [np.cos(local_moving_theta), np.sin(local_moving_theta)]
        )
        lag_distance = np.linalg.norm(swarm_center - expected_position)
        lag_threshold = 5.0
        min_slow_factor = 0.05
        max_lag_distance = 12.0
        slow_factor = 0.0 if lag_distance > max_lag_distance else max(min_slow_factor, min(1.0, lag_threshold / (lag_distance + 1e-6)))
        local_moving_theta += local_theta_step * slow_factor
        return ocm.circle_center + ocm.circle_radius * np.array(
            [np.cos(local_moving_theta), np.sin(local_moving_theta)]
        )

    # load PPO (optional)
    model = None
    if policy_zip is not None:
        if PPO is None:
            raise ImportError("stable-baselines3 is not installed, but policy_zip was provided.")
        model = PPO.load(policy_zip, device="cpu")

    # init swarm
    start_position = ocm.circle_center + ocm.circle_radius * np.array([1.0, 0.0])
    headings = np.random.uniform(0, 2 * np.pi, num_robots)
    velocities = np.zeros((num_robots, 2))
    trajectory: list[np.ndarray] = []

    if formation_type.lower() == "triangle":
        positions = ocm.initialize_positions_triangle(num_robots, start_position, ocm.formation_size_triangle_base)
    else:
        positions = ocm.initialize_positions(num_robots, start_position, ocm.formation_radius_base)

    obstacles = ocm.generate_varied_obstacles_with_levels(
        ocm.circle_center, ocm.circle_radius, ocm.num_obstacles,
        ocm.min_obstacle_size, ocm.max_obstacle_size,
        ocm.offset_degrees, ocm.passage_width, obstacle_level,
    )

    current_K, current_C = ocm.K_base, ocm.C_base
    alpha, beta = ocm.alpha_base, ocm.beta_base
    formation_radius = ocm.formation_radius_base
    formation_size_triangle = ocm.formation_size_triangle_base

    for _ in range(num_steps):
        # collisions (PSO removed; PPO should regenerate)
        positions_new, removed = ocm.check_collisions(positions, obstacles)
        positions = positions_new

        # if PPO active, REGENERATE to keep N constant
        if model is not None:
            missing = num_robots - len(positions)
            if missing > 0:
                # respawn near moving center with small jitter, zero velocity, random heading
                spawn_center = local_get_moving_center(positions if len(positions) > 0 else np.array([[ww/2, ww/2]]))
                new_pos = []
                for _ in range(missing):
                    jitter = np.random.uniform(-1.0, 1.0, size=2)
                    pos = np.clip(spawn_center + jitter, wbt, ww - wbt)
                    new_pos.append(pos)
                if len(new_pos):
                    positions = np.vstack([positions, np.array(new_pos)])
                    headings = np.concatenate([headings[:len(positions)-missing], np.random.uniform(0, 2*np.pi, size=missing)])
                    velocities = np.vstack([velocities[:len(positions)-missing], np.zeros((missing, 2))])
        else:
            # heuristic mode: actually delete the collided robots
            if removed:
                headings = np.delete(headings, removed, axis=0)
                velocities = np.delete(velocities, removed, axis=0)

        if len(positions) == 0:
            break

        # controller parameters
        if model is not None:
            obs = _build_obs(positions, headings, velocities)
            # if something still mismatches (shouldn't), pad
            expected = 5 * num_robots
            if obs.size != expected:
                pad = expected - obs.size
                if pad > 0:
                    obs = np.concatenate([obs, np.zeros(pad, dtype=np.float32)])
                else:
                    obs = obs[:expected]
            action, _ = model.predict(obs, deterministic=True)
            alpha = float(np.clip(action[0], 0.0, 1.0))
            beta  = float(np.clip(action[1], 0.0, 1.0))
            current_K = float(np.clip(action[2], ocm.K_min, ocm.K_max))
            current_C = float(np.clip(action[3], ocm.C_min, ocm.C_max))
        else:
            formation_radius, formation_size_triangle, alpha, beta = ocm.adapt_parameters(
                positions, obstacles,
                ocm.formation_radius_base, ocm.formation_size_triangle_base,
                ocm.alpha_base, ocm.beta_base,
            )

        # targets
        center_now = local_get_moving_center(positions)
        if formation_type.lower() == "triangle":
            targets = ocm.get_target_positions_triangle(center_now, len(positions), formation_size_triangle)
        else:
            targets = ocm.get_target_positions(center_now, len(positions), formation_radius)

        # dynamics
        forces, current_K, current_C = ocm.compute_forces_with_sensors(
            positions, headings, velocities, targets, obstacles,
            current_K, current_C, alpha, beta
        )
        positions, headings = ocm.update_positions_and_headings(
            positions, headings, forces, rms, (ww, wbt)
        )

        trajectory.append(positions.copy())

    return {
        "trajectory": trajectory,
        "obstacles": obstacles,
        "world_width": ww,
        "formation_type": formation_type,
        "obstacle_level": obstacle_level,
    }


def plot_trajectory_from_result(res):
    """Render a static trajectory plot and save to disk."""
    traj = res["trajectory"]
    if not traj:
        print("No trajectory to plot (all robots removed immediately?).")
        return

    ww = res["world_width"]
    obstacles = res["obstacles"]
    formation_type = res["formation_type"]
    obstacle_level = res["obstacle_level"]

    plt.figure(figsize=(8, 8))

    # Robots may be removed mid-run; plot only indices that exist at each step
    first = traj[0]
    for i in range(len(first)):
        path = np.array([step[i] for step in traj if len(step) > i])
        if len(path) >= 2:
            plt.plot(path[:, 0], path[:, 1], linewidth=1.25)

    ax = plt.gca()
    for obs in obstacles:
        if obs.get("type", "circle") == "circle":
            ax.add_patch(plt.Circle(obs["position"], obs["radius"], color="red", alpha=0.5))

    plt.title(f"Trajectory: {formation_type.capitalize()} Formation | Obstacle Level {obstacle_level}")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.xlim(0, ww)
    plt.ylim(0, ww)
    ax.set_aspect("equal")
    plt.grid(True)
    plt.tight_layout()
    fn = f"trajectory_{formation_type}_obstacle{obstacle_level}.png"
    plt.savefig(fn, dpi=150)
    plt.close()
    print(f"Saved {fn}")


def main():
    # IMPORTANT: for a trained PPO policy, keep N equal to the training size (usually 15).
    N = ocm.num_robots  # should be 15 for your trained policy
    for formation_type in ["circle", "triangle"]:
        for obstacle_level in [0, 1, 2, 3, 4]:
            print(f"Running: Formation={formation_type}, Obstacle={obstacle_level}")
            res = simulate_and_record_trajectory_cfg(
                formation_type,
                obstacle_level,
                num_robots=N,
                num_steps=8000,  # adjust as you like
                world_width_=ocm.world_width,
                robot_max_speed_=ocm.robot_max_speed,
                world_boundary_tolerance_=ocm.world_boundary_tolerance,
                policy_zip=POLICY_ZIP_PATH,  # set to None to disable PPO
            )
            plot_trajectory_from_result(res)


if __name__ == "__main__":
    main()
