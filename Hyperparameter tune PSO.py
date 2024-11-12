import numpy as np
import random
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing

# PSO function
def pso(start_checkpoint, rest_checkpoints, width, num_particles, num_iterations, inertia, cognitive_coeff, social_coeff):
    def distance(p1, p2):
        """Calculate Euclidean distance between two points."""
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def route_distance(route):
        """Calculate the total distance for a route (circular path)."""
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += distance(route[i], route[i + 1])
        total_distance += distance(route[-1], route[0])  # Return to start
        return total_distance

    class Particle:
        def __init__(self, start_checkpoint, rest_checkpoints):
            self.start_checkpoint = tuple(start_checkpoint)
            self.rest_checkpoints = [tuple(cp) for cp in rest_checkpoints]
            # Initialize route starting from start_checkpoint
            self.route = [self.start_checkpoint] + random.sample(self.rest_checkpoints, len(self.rest_checkpoints))
            self.best_route = self.route[:]
            self.best_distance = route_distance(self.route)

        def update_personal_best(self):
            current_distance = route_distance(self.route)
            if current_distance < self.best_distance:
                self.best_distance = current_distance
                self.best_route = self.route[:]

        def update_velocity(self, global_best_route):
            """PSO velocity update based on best routes (cognitive + social)"""
            new_route = self.route[:]
            # Indices of rest_checkpoints in route (excluding start_checkpoint)
            route_indices = list(range(1, len(new_route)))

            for i in route_indices:
                r1, r2 = random.random(), random.random()
                # Cognitive component
                if r1 < cognitive_coeff:
                    idx = self.best_route.index(self.route[i])
                    if idx != i:
                        new_route[i], new_route[idx] = new_route[idx], new_route[i]
                # Social component
                if r2 < social_coeff:
                    idx = global_best_route.index(self.route[i])
                    if idx != i:
                        new_route[i], new_route[idx] = new_route[idx], new_route[i]
            self.route = new_route

    # Initialize particles and global best
    particles = [Particle(start_checkpoint, rest_checkpoints) for _ in range(num_particles)]
    global_best_particle = min(particles, key=lambda p: p.best_distance)
    global_best_route = global_best_particle.best_route[:]
    global_best_distance = global_best_particle.best_distance

    # PSO main loop
    for _ in range(num_iterations):
        for particle in particles:
            particle.update_velocity(global_best_route)
            particle.update_personal_best()

        # Update global best if needed
        candidate_best = min(particles, key=lambda p: p.best_distance)
        if candidate_best.best_distance < global_best_distance:
            global_best_route = candidate_best.best_route[:]
            global_best_distance = candidate_best.best_distance

    # Return both the optimal route and total distance
    return global_best_route, global_best_distance

# Parameters (Moved outside the __main__ block)
width = 35
num_checkpoints = 8

# Generate random checkpoints with distinct start and end positions (Moved outside the __main__ block)
np.random.seed(21)  # For reproducibility
checkpoints = [tuple(np.random.rand(2) * width) for _ in range(num_checkpoints)]
start_checkpoint = checkpoints[0]
rest_checkpoints = checkpoints[1:]

def run_pso(params):
    """
    Function to run PSO with given hyperparameters.
    This will be called by each process in the pool.
    """
    num_particles, num_iterations, inertia, cognitive_coeff, social_coeff = params
    # Run PSO with current hyperparameters
    optimal_route, total_distance = pso(
        start_checkpoint=start_checkpoint,
        rest_checkpoints=rest_checkpoints,
        width=width,
        num_particles=num_particles,
        num_iterations=num_iterations,
        inertia=inertia,
        cognitive_coeff=cognitive_coeff,
        social_coeff=social_coeff
    )
    # Return the results as a dictionary
    return {
        'num_particles': num_particles,
        'num_iterations': num_iterations,
        'inertia': inertia,
        'cognitive_coeff': cognitive_coeff,
        'social_coeff': social_coeff,
        'total_distance': total_distance
    }

if __name__ == '__main__':
    # Define hyperparameter ranges
    num_particles_list = [100, 150, 200, 250]
    num_iterations_list = [100, 150, 200, 250]
    inertia_list = [0.3, 0.5, 0.7, 0.9]
    cognitive_coeff_list = [0.5, 0.75, 1.0, 1.25]
    social_coeff_list = [1.25, 1.5, 1.75, 2.0]

    # Generate all combinations
    param_grid = list(itertools.product(
        num_particles_list,
        num_iterations_list,
        inertia_list,
        cognitive_coeff_list,
        social_coeff_list
    ))

    # Total number of combinations
    total_combinations = len(param_grid)
    print(f"Total hyperparameter combinations to evaluate: {total_combinations}")

    # Use multiprocessing Pool to parallelize the computation
    with multiprocessing.Pool() as pool:
        # Map the run_pso function over all combinations
        results_list = pool.map(run_pso, param_grid)

    # Convert the results list to a DataFrame
    results = pd.DataFrame(results_list)

    # Analyze results
    best_result = results.loc[results['total_distance'].idxmin()]
    print("\nBest Hyperparameters:")
    print(best_result)

    # Optional: Display the top 5 configurations
    print("\nTop 5 Hyperparameter Configurations:")
    print(results.nsmallest(5, 'total_distance'))

    # Plot the optimized route with the best hyperparameters
    optimal_route, _ = pso(
        start_checkpoint=start_checkpoint,
        rest_checkpoints=rest_checkpoints,
        width=width,
        num_particles=int(best_result['num_particles']),
        num_iterations=int(best_result['num_iterations']),
        inertia=best_result['inertia'],
        cognitive_coeff=best_result['cognitive_coeff'],
        social_coeff=best_result['social_coeff']
    )

    # Redefine start, intermediate, and end checkpoints based on optimized route
    intermediate_checkpoints = optimal_route[1:]
    end_checkpoint = optimal_route[-1]

    # Graph the optimized route
    plt.figure(figsize=(8, 6))
    plt.plot(*zip(*optimal_route), marker='o', color='blue', label='Optimized Route')
    plt.scatter(*zip(*optimal_route), color='blue')
    plt.scatter(*zip(*checkpoints), color='gray', label='Checkpoints')
    plt.scatter(*start_checkpoint, color='green', marker='o', s=100, label='Start')
    plt.scatter(*end_checkpoint, color='red', marker='x', s=100, label='End')
    plt.scatter(*zip(*intermediate_checkpoints), color='orange', marker='s', s=60, label='Intermediate Checkpoints')
    plt.title("Optimized Route using PSO with Best Hyperparameters")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()
