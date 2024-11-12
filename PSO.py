import numpy as np
import random

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
