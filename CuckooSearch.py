import numpy as np
import random

# --- Problem Setup ---
values = [60, 100, 120]       # Example item values
weights = [10, 20, 30]        # Item weights
capacity = 50                 # Knapsack capacity
n_items = len(values)

# --- CSA Parameters ---
n_nests = 5                   # Number of nests
pa = 0.25                     # Discovery probability
max_iter = 20                 # Max iterations


# Fitness Function (total value, penalize overweight)
def fitness(solution):
    total_value = np.sum(np.array(values) * solution)
    total_weight = np.sum(np.array(weights) * solution)
    if total_weight > capacity:  # Penalize infeasible solutions
        return -1
    return total_value


# Levy Flight (generates new candidate solution)
def levy_flight(Lambda=1.5):
    u = np.random.normal(0, 1)
    v = np.random.normal(0, 1)
    step = u / (abs(v) ** (1 / Lambda))
    return step


# Generate new solution using Levy flight
def get_new_solution(sol):
    new_sol = sol.copy()
    for i in range(n_items):
        if random.random() < 0.5:  # Apply random mutation
            new_sol[i] = 1 - new_sol[i]  # Flip 0↔1
    return new_sol


# --- Initialization: random nests ---
nests = [np.random.randint(0, 2, n_items) for _ in range(n_nests)]
fitness_values = [fitness(sol) for sol in nests]

best_idx = np.argmax(fitness_values)
best_solution = nests[best_idx]
best_value = fitness_values[best_idx]

print("Initial Best:", best_solution, "Value:", best_value)

# --- CSA Main Loop ---
for gen in range(max_iter):
    # Generate new solutions (cuckoo lays eggs)
    for i in range(n_nests):
        new_sol = get_new_solution(nests[i])
        new_fit = fitness(new_sol)
        if new_fit > fitness_values[i]:
            nests[i] = new_sol
            fitness_values[i] = new_fit

    # Abandon worst nests with probability pa
    for i in range(n_nests):
        if random.random() < pa:
            nests[i] = np.random.randint(0, 2, n_items)
            fitness_values[i] = fitness(nests[i])

    # Update best solution
    best_idx = np.argmax(fitness_values)
    if fitness_values[best_idx] > best_value:
        best_solution = nests[best_idx]
        best_value = fitness_values[best_idx]

    print(f"Iteration {gen+1}: Best = {best_solution}, Value = {best_value}")

print("\nFinal Best Solution:", best_solution)
print("Final Best Value:", best_value)
