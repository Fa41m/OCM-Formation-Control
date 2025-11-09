# trajectory_runner.py
import numpy as np
import matplotlib.pyplot as plt
import ocm  # your module containing the sim primitives & constants


def simulate_and_record_trajectory_cfg(
    formation_type: str,
    obstacle_level: int,
    *,
    num_robots: int = 15,
    num_steps: int = 400,
    world_width_: float = None,
    robot_max_speed_: float = None,
    world_boundary_tolerance_: float = None,
):
    """
    Runs a non-animated simulation and records positions over time for trajectory plots.
    Fully parameterized (no unqualified globals).
    """
    # Derive environment/dynamics params from ocm if not provided
    ww = ocm.world_width if world_width_ is None else world_width_
    rms = ocm.robot_max_speed if robot_max_speed_ is None else robot_max_speed_
    wbt = (
        ocm.world_boundary_tolerance
        if world_boundary_tolerance_ is None
        else world_boundary_tolerance_
    )

    # Local moving-center state (decoupled from any prior animation state)
    local_moving_theta = 0.0
    local_theta_step = 2.0 * np.pi / float(num_steps)

    def local_get_moving_center(frame, swarm_positions):
        """Copy of your logic, but with local angle state so each run is independent."""
        nonlocal local_moving_theta

        swarm_center = np.mean(swarm_positions, axis=0)
        expected_position = ocm.circle_center + ocm.circle_radius * np.array(
            [np.cos(local_moving_theta), np.sin(local_moving_theta)]
        )
        lag_distance = np.linalg.norm(swarm_center - expected_position)

        lag_threshold = 5.0
        min_slow_factor = 0.05
        max_lag_distance = 12.0

        if lag_distance > max_lag_distance:
            slow_factor = 0.0
        else:
            slow_factor = max(
                min_slow_factor, min(1.0, lag_threshold / (lag_distance + 1e-6))
            )

        local_moving_theta += local_theta_step * slow_factor
        return ocm.circle_center + ocm.circle_radius * np.array(
            [np.cos(local_moving_theta), np.sin(local_moving_theta)]
        )

    # ---- Initialize swarm ----
    start_position = ocm.circle_center + ocm.circle_radius * np.array([1.0, 0.0])
    headings = np.random.uniform(0, 2 * np.pi, num_robots)
    velocities = np.zeros((num_robots, 2))
    trajectory = []  # list of (N_i x 2) arrays over time

    if formation_type.lower() == "triangle":
        positions = ocm.initialize_positions_triangle(
            num_robots, start_position, ocm.formation_size_triangle_base
        )
    else:
        positions = ocm.initialize_positions(
            num_robots, start_position, ocm.formation_radius_base
        )

    # Obstacles
    obstacles = ocm.generate_varied_obstacles_with_levels(
        ocm.circle_center,
        ocm.circle_radius,
        ocm.num_obstacles,
        ocm.min_obstacle_size,
        ocm.max_obstacle_size,
        ocm.offset_degrees,
        ocm.passage_width,
        obstacle_level,
    )

    # Adaptive params current state
    current_K, current_C = ocm.K_base, ocm.C_base
    alpha, beta = ocm.alpha_base, ocm.beta_base
    formation_radius = ocm.formation_radius_base
    formation_size_triangle = ocm.formation_size_triangle_base

    # ---- Sim loop ----
    for frame in range(num_steps):
        positions_new, removed = ocm.check_collisions(positions, obstacles)
        positions = positions_new
        if len(positions) == 0:
            break
        if removed:
            headings = np.delete(headings, removed, axis=0)
            velocities = np.delete(velocities, removed, axis=0)

        # Adapt formation based on environment density/proximity
        formation_radius, formation_size_triangle, alpha, beta = ocm.adapt_parameters(
            positions,
            obstacles,
            ocm.formation_radius_base,
            ocm.formation_size_triangle_base,
            ocm.alpha_base,
            ocm.beta_base,
        )

        # Targets on moving path
        center_now = local_get_moving_center(frame, positions)
        if formation_type.lower() == "triangle":
            targets = ocm.get_target_positions_triangle(
                center_now, len(positions), formation_size_triangle
            )
        else:
            targets = ocm.get_target_positions(
                center_now, len(positions), formation_radius
            )

        # Forces & dynamics
        forces, current_K, current_C = ocm.compute_forces_with_sensors(
            positions,
            headings,
            velocities,
            targets,
            obstacles,
            current_K,
            current_C,
            alpha,
            beta,
        )

        positions, headings = ocm.update_positions_and_headings(
            positions, headings, forces, rms, (ww, wbt)
        )

        # Store snapshot
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
    # Draw obstacles
    for obs in obstacles:
        if obs.get("type", "circle") == "circle":
            ax.add_patch(
                plt.Circle(obs["position"], obs["radius"], color="red", alpha=0.5)
            )

    plt.title(
        f"Trajectory: {formation_type.capitalize()} Formation | Obstacle Level {obstacle_level}"
    )
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
    # Use your ocm values by default, but everything is overrideable
    for formation_type in ["circle", "triangle"]:
        for obstacle_level in [0, 1, 2, 3, 4]:
            print(f"Running: Formation={formation_type}, Obstacle={obstacle_level}")
            res = simulate_and_record_trajectory_cfg(
                formation_type,
                obstacle_level,
                num_robots=ocm.num_robots,
                # num_steps=ocm.num_steps,  # or use a shorter run like 400 if you prefer
                num_steps=1600,  # or use a shorter run like 400 if you prefer
                world_width_=ocm.world_width,
                robot_max_speed_=ocm.robot_max_speed,
                world_boundary_tolerance_=ocm.world_boundary_tolerance,
            )
            plot_trajectory_from_result(res)


if __name__ == "__main__":
    main()
