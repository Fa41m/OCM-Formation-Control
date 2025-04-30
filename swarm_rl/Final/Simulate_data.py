import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import time

# Import your existing code
import ocm
import train_and_simulate
import env
from ocm import *
from train_and_simulate import *
from env import *

# Setting up the directory structure for data storage

DATA_ROOT = "Data"
PSO_ROOT  = os.path.join(DATA_ROOT, "PSO")
RL_ROOT  = os.path.join(DATA_ROOT, "RL")
OBSTACLE_LEVEL_DIRS = [f"obstacle_level_{i}" for i in range(5)]

# Create directories if they don't exist
def ensure_directory_structure():
    if not os.path.exists(DATA_ROOT):
        os.mkdir(DATA_ROOT)
    if not os.path.exists(PSO_ROOT):
        os.mkdir(PSO_ROOT)
    for d in OBSTACLE_LEVEL_DIRS:
        dir_path = os.path.join(PSO_ROOT, d)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
    if not os.path.exists(RL_ROOT):
        os.mkdir(RL_ROOT)
    for d in OBSTACLE_LEVEL_DIRS:
        dir_path = os.path.join(RL_ROOT, d)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
    

# Original run_simulation for a single final cost
def run_simulation(num_robots_test, obstacle_level_test, alpha, beta, K, C, do_plot=False):
    global num_robots, obstacle_level, alpha_base, beta_base, K_base, C_base, robot_max_speed
    global total_cost, world_width, world_boundary_tolerance, sensor_detection_distance
    global circle_center, circle_radius, formation_type, num_steps
    num_robots = num_robots_test
    obstacle_level = obstacle_level_test
    alpha_base = alpha
    beta_base  = beta
    K_base     = K
    C_base     = C
    world_width = num_robots * 4
    robot_max_speed = 0.3  

    # Reset cost tracking
    global total_cost
    total_cost = 0.0

    # Initialize the swarm
    angles = np.random.uniform(0, 2 * np.pi, num_robots)
    positions = np.array([
        circle_center + circle_radius * np.array([np.cos(ang), np.sin(ang)]) 
        for ang in angles
    ])
    headings = np.random.uniform(0, 2 * np.pi, num_robots)
    velocities = np.zeros((num_robots, 2))

    obstacles = generate_varied_obstacles_with_levels(
        circle_center,
        circle_radius,
        3,   # num_obstacles
        world_width / 50,
        world_width / 25,
        50,  # offset_degrees
        world_width / 15,
        obstacle_level
    )

    current_K = K_base
    current_C = C_base
    formation_radius_base = num_robots // 5
    formation_size_triangle_base = num_robots // 10
    formation_radius = formation_radius_base
    formation_size_triangle = formation_size_triangle_base

    # Tracking
    K_values_over_time = [current_K]
    C_values_over_time = [current_C]
    alpha_values_over_time = [alpha_base]
    beta_values_over_time = [beta_base]
    cost_history = [0.0]
    
    total_steps = 1600

    # 4) Main simulation loop
    for frame in range(total_steps):
        # A) Collisions
        updated_positions, removed_indices = check_collisions(positions, obstacles)
        if removed_indices:
            headings = np.delete(headings, removed_indices, axis=0)
            velocities = np.delete(velocities, removed_indices, axis=0)
            positions = updated_positions
        else:
            positions = updated_positions

        if len(positions) == 0:
            # All robots removed => break
            break

        # B) Adaptive parameters
        formation_radius, formation_size_triangle, alpha_curr, beta_curr = adapt_parameters(
            positions, obstacles,
            formation_radius_base, formation_size_triangle_base,
            alpha_base, beta_base
        )

        # C) Generate target positions
        moving_center = get_moving_center(frame, num_steps, positions)
        if formation_type.lower() == "triangle":
            target_positions = get_target_positions_triangle(moving_center, len(positions), formation_size_triangle)
        else:
            target_positions = get_target_positions(moving_center, len(positions), formation_radius)

        # D) Compute forces & update K, C
        forces, current_K, current_C = compute_forces_with_sensors(
            positions,
            headings,
            velocities,
            target_positions,
            obstacles,
            current_K,
            current_C,
            alpha_curr,
            beta_curr
        )

        # E) Update positions & headings
        positions, headings = update_positions_and_headings(
            positions,
            headings,
            forces,
            robot_max_speed,
            (world_width, world_boundary_tolerance)
        )

        # F) Cost function
        psi = ocm.compute_swarm_alignment(headings)
        alignment_cost = ocm.cost_w_align * (1.0 - psi)**2

        path_errors = np.linalg.norm(positions - target_positions, axis=1)
        avg_path_error = np.mean(path_errors) if len(path_errors) > 0 else 0.0
        path_cost = ocm.cost_w_path * avg_path_error

        collision_cost = 0.0
        safe_distance = sensor_detection_distance / 2.0
        for i in range(len(positions)):
            for obs in obstacles:
                obs_dist = np.linalg.norm(positions[i] - obs["position"]) - obs["radius"]
                if obs_dist < safe_distance:
                    collision_cost += ocm.cost_w_obs * np.exp(-obs_dist)

        sum_force = np.sum(np.linalg.norm(forces, axis=1))
        control_cost = ocm.cost_w_force * sum_force

        cost_step = alignment_cost + path_cost + collision_cost + control_cost
        total_cost += cost_step

        # G) Logging for optional plots
        K_values_over_time.append(current_K)
        C_values_over_time.append(current_C)
        alpha_values_over_time.append(alpha_curr)
        beta_values_over_time.append(beta_curr)
        cost_history.append(total_cost)

    # Optional plotting
    if do_plot:
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

    return total_cost

# Modified run_simulation for logging per-timestep cost details
def run_simulation_with_logging(num_robots_test, obstacle_level_test, alpha, beta, K, C):
    global num_robots, obstacle_level, alpha_base, beta_base, K_base, C_base
    num_robots = num_robots_test
    obstacle_level = obstacle_level_test
    alpha_base = alpha
    beta_base  = beta
    K_base     = K
    C_base     = C

    global total_cost
    total_cost = 0.0

    angles = np.random.uniform(0, 2 * np.pi, num_robots)
    positions = np.array([
        circle_center + circle_radius * np.array([np.cos(ang), np.sin(ang)]) 
        for ang in angles
    ])
    headings = np.random.uniform(0, 2 * np.pi, num_robots)
    velocities = np.zeros((num_robots, 2))

    obstacles = generate_varied_obstacles_with_levels(
        circle_center,
        circle_radius,
        3,   
        world_width / 50,
        world_width / 25,
        50, 
        world_width / 15,
        obstacle_level
    )

    current_K = K_base
    current_C = C_base
    formation_radius_base = num_robots // 5
    formation_size_triangle_base = num_robots // 10

    # List to store per-timestep cost data
    timestep_data = []
    collisions = 0
    
    total_steps = num_steps*2

    for frame in range(total_steps):
        updated_positions, removed_indices = check_collisions(positions, obstacles)
        if removed_indices:
            collisions += len(removed_indices)
            headings = np.delete(headings, removed_indices, axis=0)
            velocities = np.delete(velocities, removed_indices, axis=0)
            positions = updated_positions
        else:
            positions = updated_positions

        if len(positions) == 0:
            break

        formation_radius, formation_size_triangle, alpha_curr, beta_curr = adapt_parameters(
            positions, obstacles,
            formation_radius_base, formation_size_triangle_base,
            alpha_base, beta_base
        )

        moving_center = get_moving_center(frame, num_steps, positions)
        if formation_type.lower() == "triangle":
            target_positions = get_target_positions_triangle(moving_center, len(positions), formation_size_triangle)
        else:
            target_positions = get_target_positions(moving_center, len(positions), formation_radius)

        forces, current_K, current_C = compute_forces_with_sensors(
            positions,
            headings,
            velocities,
            target_positions,
            obstacles,
            current_K,
            current_C,
            alpha_curr,
            beta_curr
        )

        positions, headings = update_positions_and_headings(
            positions,
            headings,
            forces,
            robot_max_speed,
            (world_width, world_boundary_tolerance)
        )

        # Cost Calculation
        psi = ocm.compute_swarm_alignment(headings)
        alignment_cost = ocm.cost_w_align * (1.0 - psi)**2

        path_errors = np.linalg.norm(positions - target_positions, axis=1)
        avg_path_error = np.mean(path_errors) if len(path_errors) > 0 else 0.0
        path_cost = ocm.cost_w_path * avg_path_error

        collision_cost = 0.0
        safe_distance = sensor_detection_distance / 2.0
        for i in range(len(positions)):
            for obs in obstacles:
                obs_dist = np.linalg.norm(positions[i] - obs["position"]) - obs["radius"]
                if obs_dist < safe_distance:
                    collision_cost += ocm.cost_w_obs * np.exp(-obs_dist)

        sum_force = np.sum(np.linalg.norm(forces, axis=1))
        control_cost = ocm.cost_w_force * sum_force

        cost_step = alignment_cost + path_cost + collision_cost + control_cost
        total_cost += cost_step

        # Store data for this timestep
        timestep_data.append([
            frame, 
            alpha_curr, 
            beta_curr, 
            current_K, 
            current_C,
            alignment_cost, 
            path_cost, 
            collision_cost, 
            control_cost, 
            total_cost,
            collisions
        ])

    return timestep_data

# PSO Optimization 
def pso_optimize(swarm_size, obstacle_level_test, n_particles=25, n_iterations=10, alpha_bounds=(0.1, 1.0), beta_bounds=(0.1, 1.0), K_bounds=(0.1, 0.99), C_bounds=(0.1, 0.99), w_max=1.2, w_min=0.4, c1=2.05, c2=2.05, cv_threshold=0.05, cv_window=20):
    # Initialize swarm
    swarm_pos = np.random.uniform(
        [alpha_bounds[0], beta_bounds[0], K_bounds[0], C_bounds[0]],
        [alpha_bounds[1], beta_bounds[1], K_bounds[1], C_bounds[1]],
        (n_particles, 4)
    )
    swarm_vel = np.random.uniform(-0.01, 0.01, (n_particles, 4))

    # Initialize personal best
    personal_best_positions = swarm_pos.copy()
    personal_best_costs = np.zeros(n_particles)

    for i in range(n_particles):
        # Evaluate cost
        personal_best_costs[i] = run_simulation(
            num_robots_test=20,
            obstacle_level_test=obstacle_level_test,
            alpha=swarm_pos[i,0],
            beta=swarm_pos[i,1],
            K=swarm_pos[i,2],
            C=swarm_pos[i,3],
            do_plot=False
        )

    # Initialize global best from personal bests 
    global_best_idx = np.argmin(personal_best_costs)
    global_best_position = personal_best_positions[global_best_idx].copy()
    global_best_cost = personal_best_costs[global_best_idx]

    recent_best_costs = []

    # PSO loop
    for iteration in range(n_iterations):
        w = w_max - ((w_max - w_min) * (iteration / n_iterations))

        for i in range(n_particles):
            r1 = np.random.rand(4)
            r2 = np.random.rand(4)

            phi = c1 + c2
            k = 2 / abs(2 - phi - np.sqrt(phi ** 2 - 4 * phi))

            swarm_vel[i] = k * (
                w * swarm_vel[i]
                + c1 * r1 * (personal_best_positions[i] - swarm_pos[i])
                + c2 * r2 * (global_best_position - swarm_pos[i])
            )

            swarm_pos[i] += swarm_vel[i]
            swarm_pos[i] = np.clip(
                swarm_pos[i],
                [alpha_bounds[0], beta_bounds[0], K_bounds[0], C_bounds[0]],
                [alpha_bounds[1], beta_bounds[1], K_bounds[1], C_bounds[1]]
            )

            # Monte Carlo averaging
            num_mc_runs = 3
            costs = []
            for _ in range(num_mc_runs):
                c_val = run_simulation(
                    num_robots_test=swarm_size,
                    obstacle_level_test=obstacle_level_test,
                    alpha=swarm_pos[i,0],
                    beta=swarm_pos[i,1],
                    K=swarm_pos[i,2],
                    C=swarm_pos[i,3],
                    do_plot=False
                )
                costs.append(c_val)
            avg_cost = np.mean(costs)

            if avg_cost < personal_best_costs[i]:
                personal_best_costs[i] = avg_cost
                personal_best_positions[i] = swarm_pos[i].copy()

                if avg_cost < global_best_cost:
                    global_best_cost = avg_cost
                    global_best_position = swarm_pos[i].copy()

        recent_best_costs.append(global_best_cost)
        # Maintain a rolling window of recent best costs
        if len(recent_best_costs) > cv_window:
            recent_best_costs.pop(0)

        # Check if it has reached the amount requuired to check for convergence
        if len(recent_best_costs) == cv_window:
            cost_std = np.std(recent_best_costs)
            cost_mean = np.mean(recent_best_costs)
            cv = cost_std / (cost_mean + 1e-6)
            # Check for convergence threshold
            if cv < cv_threshold:
                print(f"[Obstacle {obstacle_level_test}] Converged at iteration {iteration+1} (CV={cv:.4f})")
                break

        print(f"[Obstacle {obstacle_level_test}] Iteration {iteration+1}/{n_iterations}, Best Cost: {global_best_cost:.4f}")

    return global_best_position, global_best_cost

# Function to run the heuristic simulation
def run_heuristic_simulation():
    start_time = time.time()
    print("Running heuristic simulation...")
    # For each obstacle level and swarm size, run PSO to find best parameters and log detailed cost info
    optimal_params_per_obstacle = {}
    swarm_sizes = [10, 14, 15, 16, 20]

    for level in range(0,1):
        optimal_params_per_swarm_size = {}
        for s_size in swarm_sizes:
            print(f"\n=== Running PSO for obstacle_level_{level} with swarm_size={s_size} ===")
            best_params, best_cost = pso_optimize(
                swarm_size=s_size,
                obstacle_level_test=level,
                n_particles=10,        
                n_iterations=10,      
                alpha_bounds=(0.001, 1.0),
                beta_bounds=(0.001, 1.0),
                K_bounds=(0.005, 0.995),
                C_bounds=(0.005, 0.995),
                w_max=1.2,
                w_min=0.4,
                c1=2.05,
                c2=2.05,
                cv_threshold=0.05,
                cv_window=10
            )
            elapsed_time = time.time() - start_time
            print(f"[Obstacle {level}, Swarm {s_size}] Best Params: {best_params}, Best Cost: {best_cost}, Time Elapsed: {elapsed_time:.2f} seconds")
            optimal_params_per_swarm_size[s_size] = best_params

            # Store the best params in a CSV or text file
            out_dir = os.path.join(PSO_ROOT, f"obstacle_level_{level}")
            params_file = os.path.join(out_dir, f"best_params_swarm_{s_size}.csv")
            with open(params_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["alpha", "beta", "K", "C", "best_cost"])
                writer.writerow([best_params[0], best_params[1], best_params[2], best_params[3], best_cost])

            # Using the optimal parameters, run detailed cost logging for each swarm size 5 times
            for run in range(5):
                print(f"\n[Obstacle {level}, Swarm {s_size}] Logging run {run+1} with optimal params...")
                # Run and get per-timestep data
                timestep_data = run_simulation_with_logging(
                    num_robots_test=s_size,
                    obstacle_level_test=level,
                    alpha=best_params[0],
                    beta=best_params[1],
                    K=best_params[2],
                    C=best_params[3]
                )
                # Store in CSV
                csv_filename = os.path.join(out_dir, f"optimal_run_swarm_{s_size}_run_{run+1}.csv")
                with open(csv_filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestep", "alpha", "beta", "K", "C",
                        "alignment_cost", "path_cost", "collision_cost",
                        "control_cost", "total_cost", "collisions"
                    ])
                    writer.writerows(timestep_data)

        optimal_params_per_obstacle[level] = optimal_params_per_swarm_size

    print("\nAll experiments completed. Data logged in the Data/PSO/* directories.")
    total_elapsed_time = time.time() - start_time
    print(f"Total Time Elapsed: {total_elapsed_time:.2f} seconds")

# Main function to run the simulations
def main():
    ensure_directory_structure()
    run_heuristic_simulation()

if __name__ == "__main__":
    main()
