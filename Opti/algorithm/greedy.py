import numpy as np
import random

# Parameters
n_services = 10  # Number of services
max_software = n_services  # Max number of software
POP = 20  # Number of top combinations to consider

# Resource Utilization Efficiency (RUE) calculation
def calc_RUE(matrix, software_count, r_add):
    redundancy = [1] * software_count  # Start with all redundancy levels set to 1
    # Increase redundancy level for one software to calculate the cost efficiency
    for i in range(software_count):
        redundancy[i] += 1  
        break  # Only one software's redundancy level is increased

    availability = 1.0
    for j in range(n_services):
        software_index = np.argmax(matrix[j])  # Find the software where service j is implemented
        availability *= (1 - (1 - 0.99) ** redundancy[software_index])

    RUE = availability / (software_count + r_add)
    return RUE

# Initial random assignment biased by r_add
def initialize_matrix(r_add):
    software_count = max(1, int(max_software / (1 + r_add)))  # Adjust software count based on r_add
    matrix = np.zeros((n_services, software_count))

    # Randomly assign services to software, ensuring services are contiguous
    start_service = 0
    for s in range(software_count):
        end_service = min(start_service + random.randint(1, n_services // software_count), n_services)
        matrix[start_service:end_service, s] = 1
        start_service = end_service

    return matrix, software_count

# Exploration by increasing software count
def explore_increase_software(matrix, software_count):
    if software_count >= max_software:
        return None, None

    new_software_count = software_count + 1
    new_matrix = np.zeros((n_services, new_software_count))

    # Copy the existing assignment
    new_matrix[:, :software_count] = matrix

    # Randomly assign remaining services to the new software
    remaining_services = np.where(np.sum(matrix, axis=1) == 0)[0]
    if len(remaining_services) > 0:
        start_service = random.choice(remaining_services)
        new_matrix[start_service:, new_software_count - 1] = 1

    return new_matrix, new_software_count

# Exploration by changing service implementation
def explore_change_service(matrix, software_count):
    new_matrix = np.copy(matrix)
    service_to_change = random.randint(0, n_services - 1)

    current_software = np.argmax(matrix[service_to_change])
    possible_software = list(range(software_count))
    possible_software.remove(current_software)
    new_software = random.choice(possible_software)

    new_matrix[service_to_change, current_software] = 0
    new_matrix[service_to_change, new_software] = 1

    return new_matrix, software_count

# Multi-start greedy search
def multi_start_greedy(r_add, starts=10):
    best_RUE = -float('inf')
    best_matrix = None
    best_software_count = None

    for _ in range(starts):
        matrix, software_count = initialize_matrix(r_add)
        current_RUE = calc_RUE(matrix, software_count, r_add)

        while True:
            # Explore by increasing software count
            new_matrix, new_software_count = explore_increase_software(matrix, software_count)
            if new_matrix is not None:
                new_RUE = calc_RUE(new_matrix, new_software_count, r_add)
                if new_RUE > current_RUE:
                    matrix, software_count, current_RUE = new_matrix, new_software_count, new_RUE
                    continue

            # Explore by changing service implementation
            new_matrix, new_software_count = explore_change_service(matrix, software_count)
            new_RUE = calc_RUE(new_matrix, new_software_count, r_add)
            if new_RUE > current_RUE:
                matrix, software_count, current_RUE = new_matrix, new_software_count, new_RUE
            else:
                break  # Stop when no improvement is found

        if current_RUE > best_RUE:
            best_RUE, best_matrix, best_software_count = current_RUE, matrix, software_count

    return best_matrix, best_software_count, best_RUE

# Example usage
r_add = 0.5  # Example value for r_add
best_matrix, best_software_count, best_RUE = multi_start_greedy(r_add)
print(f"Best Matrix:\n{best_matrix}")
print(f"Best Software Count: {best_software_count}")
print(f"Best RUE: {best_RUE}")
