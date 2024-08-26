import numpy as np
import random
import matplotlib.pyplot as plt

def calc_software_av(services_in_sw, service_avail, server_avail):
    non_zero_idx = np.nonzero(services_in_sw)[0]
    sw_avail = np.prod([service_avail[k] for k in non_zero_idx])
    return sw_avail * server_avail

def make_matrix(service, software_count):
    matrix = np.zeros((software_count, len(service)), dtype=int)
    splits = sorted(random.sample(range(1, len(service)), software_count - 1))
    splits.append(len(service))
    idx = 0
    for i, split in enumerate(splits):
        for j in range(idx, split):
            matrix[i][j] = 1
        idx = split
    return matrix

def calc_RUE(matrix, software_count, service_avail, server_avail, r_add, H):
    initial_redundancy = [1] * software_count
    redundancy_cost_efficiency = []
    software_availability = [calc_software_av(matrix[j], service_avail, server_avail) for j in range(software_count)]
    system_avail = np.prod([1 - (1 - sa) ** int(r) for sa, r in zip(software_availability, initial_redundancy)])
    total_servers = sum(initial_redundancy[j] * ((r_add * (np.sum(matrix[j]) - 1)) + 1) for j in range(software_count))

    for i in range(software_count):
        initial_redundancy[i] = 2
        total_servers_red = sum(initial_redundancy[j] * ((r_add * (np.sum(matrix[j]) - 1)) + 1) for j in range(software_count))

        if total_servers_red <= H:
            system_avail_red = np.prod([1 - (1 - sa) ** int(r) for sa, r in zip(software_availability, initial_redundancy)])
            redundancy_cost_efficiency.append((system_avail_red - system_avail) / (total_servers_red - total_servers))
        else:
            redundancy_cost_efficiency.append(0)

        initial_redundancy[i] = 1

    avg_efficiency = np.mean(redundancy_cost_efficiency)

    return avg_efficiency

def greedy_search(matrix, software_count, service_avail, server_avail, r_add, H, service):
    list = []
    best_matrix = matrix.copy()
    best_RUE = calc_RUE(matrix, software_count, service_avail, server_avail, r_add, H)

    for k in range(GENERATION):
        max_RUE_list = []
        for j in range(software_count):
            new_matrix = matrix.copy()
            new_matrix[j] = np.roll(new_matrix[j], 1)
            new_RUE = calc_RUE(new_matrix, software_count, service_avail, server_avail, r_add, H)
            max_RUE_list.append(new_RUE)

        if software_count < len(service):
            new_sw_p_matrix = make_matrix(service, software_count + 1)
            new_RUE_p = calc_RUE(new_sw_p_matrix, software_count + 1, service_avail, server_avail, r_add, H)
            max_RUE_list.append(new_RUE_p)

        if software_count > 1:
            new_sw_n_matrix = make_matrix(service, software_count - 1)
            new_RUE_n = calc_RUE(new_sw_n_matrix, software_count - 1, service_avail, server_avail, r_add, H)
            max_RUE_list.append(new_RUE_n)

        max_RUE = max(max_RUE_list)
        if max_RUE > best_RUE:
            if max_RUE == new_RUE:
                best_RUE = new_RUE
                best_matrix = new_matrix
            elif max_RUE == new_RUE_p:
                best_RUE = max_RUE
                best_matrix = new_sw_p_matrix
            elif max_RUE == new_RUE_n:
                best_RUE = max_RUE
                best_matrix = new_sw_n_matrix
        list.append(best_RUE)

    return best_matrix, software_count, best_RUE, list

def multi_start_greedy(r_add, service_avail, server_avail, H, num_service, num_starts=10):
    best_global_matrix = None
    best_global_RUE = -np.inf
    best_global_software_count = 0
    RUE_list = []
    x_gene = [i for i in range(1,GENERATION+1)]
    service = [i for i in range(1, num_service + 1)]

    for _ in range(num_starts):
        software_count = np.random.randint(1, num_service + 1)
        matrix = make_matrix(service, software_count)
        best_matrix, best_software_count, best_RUE, RUE_each_list = greedy_search(matrix, software_count, service_avail, server_avail, r_add, H, service)
        RUE_list.extend(RUE_each_list)

        if best_RUE > best_global_RUE:
            best_global_RUE = best_RUE
            best_global_matrix = best_matrix
            best_global_software_count = best_software_count

        plt.plot(x_gene, RUE_each_list)
    return best_global_matrix, best_global_software_count, best_global_RUE

# 使用例
r_add = 1
num_service = 10
service_avail = [0.99] * num_service
server_avail = 0.99
H = 20

GENERATION = 20

fig, ax = plt.subplots(figsize=(12, 8))
best_matrix, best_software_count, best_RUE = multi_start_greedy(r_add, service_avail, server_avail, H, num_service)

plt.xlabel("Generation")
plt.ylabel("RCE")
plt.show()

print(f"Best Matrix:\n{best_matrix}")
print(f"Best Software Count: {best_software_count}")
print(f"Best RCE: {best_RUE}")
