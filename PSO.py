import numpy as np
import random

def pso(checkpoints, num_particles, num_iterations, inertia, cognitive_coeff, social_coeff):

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
        def __init__(self, checkpoints):
            self.checkpoints = [tuple(cp) for cp in checkpoints]  # Convert checkpoints to tuples
            self.route = random.sample(self.checkpoints, len(self.checkpoints))
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
            for i in range(len(self.route)):
                if random.random() < cognitive_coeff:
                    # Swap current route element with personal best
                    idx = self.best_route.index(self.route[i])
                    new_route[i], new_route[idx] = new_route[idx], new_route[i]
                elif random.random() < social_coeff:
                    # Swap current route element with global best
                    idx = global_best_route.index(self.route[i])
                    new_route[i], new_route[idx] = new_route[idx], new_route[i]
            self.route = new_route

    # Initialize particles and global best
    checkpoints = [tuple(cp) for cp in checkpoints]  # Convert checkpoints to tuples
    particles = [Particle(checkpoints) for _ in range(num_particles)]
    global_best_route = min(particles, key=lambda p: p.best_distance).best_route
    global_best_distance = route_distance(global_best_route)

    # PSO main loop
    for _ in range(num_iterations):
        for particle in particles:
            particle.update_personal_best()
            particle.update_velocity(global_best_route)
        
        # Update global best if needed
        candidate_best = min(particles, key=lambda p: p.best_distance)
        if candidate_best.best_distance < global_best_distance:
            global_best_route = candidate_best.best_route
            global_best_distance = candidate_best.best_distance

    # The optimal checkpoint sequence determined by PSO
    print("Optimal route order by PSO:", global_best_route)
    print("Total distance:", global_best_distance)
    
    return global_best_route

# Test Cases
# checkpoints = [(0, 0), (2, 3), (5, 5), (8, 8), (6, 1)]
# pso(checkpoints, num_particles=30, num_iterations=100, inertia=0.5, cognitive_coeff=1.5, social_coeff=1.5)