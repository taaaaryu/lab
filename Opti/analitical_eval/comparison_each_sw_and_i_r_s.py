import time
import matplotlib.pyplot as plt
import math
import numpy as np
from itertools import combinations, chain, product
import ast
import japanize_matplotlib
from sklearn import preprocessing  # 正規化用
import csv

# Parameters
h_adds = [0.8]  # Increment in server count per additional service
POP = 0.1  # Top combinations to consider

# Constants
N = [9,11,13]  # Number of services

server_avail = 0.99
max_redundancy = 5

# Function to calculate software availability
def calc_software_av(services_group, service_avail):
    indices = [services.index(s) for s in services_group]
    return np.prod([service_avail[i] for i in indices])

# Function to generate service combinations
def generate_service_combinations(services, num_software):
    all_combinations = []
    n = len(services)
    for indices in combinations(range(n - 1), num_software - 1):
        split_indices = list(chain([-1], indices, [n - 1]))
        combination = [services[split_indices[i] + 1: split_indices[i + 1] + 1] for i in range(len(split_indices) - 1)]
        all_combinations.append(combination)
    return all_combinations

# Function to generate redundancy combinations
def generate_redundancy_combinations(num_software, max_servers, h_add):
    all_redundancies = [redundancy for redundancy in product(range(1, max_redundancy), repeat=num_software)]
    return all_redundancies

# Function to search for the best redundancy
def search_best_redundancy(all_combinations, all_redundancies, RCE):
    r_unav = []
    best_reds = []
    best_comb = []
    exist_RCE = []  # RCEの高さとavailabilityの相関比較用
    for k in range(len(all_combinations)):
        comb = all_combinations[k]
        each_redundancy = all_redundancies[len(comb) - 1]
        max_system_avail = -1
        best_redundancy = None
        for redundancy in each_redundancy:
            total_servers = sum(redundancy[i] * (len(comb[i]) * (h_add ** (len(comb[i]) - 1))) for i in range(len(comb)))
            if total_servers <= H:
                if alloc <= total_servers:
                    software_availability = [calc_software_av(group, service_avail) * server_avail for group in comb]
                    system_avail = np.prod([1 - (1 - sa) ** int(r) for sa, r in zip(software_availability, redundancy)])
                    if system_avail > max_system_avail:
                        max_system_avail = system_avail
                        best_redundancy = redundancy
        if best_redundancy:
            r_unav.append(1 - max_system_avail)
            best_reds.append(best_redundancy)
            best_comb.append(comb)
            exist_RCE.append(RCE[k])

    return r_unav, best_reds, best_comb, exist_RCE

# CSV出力の関数
def save_results_to_csv(filename, p_comb, p_rue, unav, red, comb, before_red_RCE):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # ヘッダー行
        writer.writerow(["Combination", "RCE", "Unavailability", "Best Redundancy", "Best Combination", "Before Redundancy RCE"])
        
        # データ行の書き込み
        for i in range(len(unav)):
            writer.writerow([
                p_comb[i] if i < len(p_comb) else "N/A",
                p_rue[i] if i < len(p_rue) else "N/A",
                unav[i] if i < len(unav) else "N/A",
                red[i] if i < len(red) else "N/A",
                comb[i] if i < len(comb) else "N/A",
                before_red_RCE[i] if i < len(before_red_RCE) else "N/A"
            ])

fig, ax = plt.subplots(figsize=(12, 8))

# Main process
for n in N:
    softwares = [i for i in range(1, n + 1)]
    services = [i for i in range(1, n + 1)]
    service_avail = [0.99] * n
    H = n * 2
    alloc = H * 0.9  # Minimum server resource allocation

    for h_add in h_adds:
        start = time.time()
        
        # Calculate and plot CDF for system availability after redundancy
        p_comb = []
        p_rue = []
        all_red = []
        for num_software in softwares:
            all_combinations = generate_service_combinations(services, num_software)
            sw_redundancies = generate_redundancy_combinations(num_software, H, h_add)
            for comb in all_combinations:
                max_system_avail = None
                total_servers = sum(len(comb[i]) * (h_add ** (len(comb[i]) - 1)) for i in range(len(comb)))
                if total_servers <= H:
                    software_availability = [calc_software_av(group, service_avail) * server_avail for group in comb]
                    system_avail = np.prod([sa for sa in software_availability])

                    # Set initial redundancy to 1 for all and then increase for one software
                    initial_redundancy = [1] * len(comb)
                    redundancy_cost_efficiency = []

                    for i in range(len(comb)):
                        initial_redundancy[i] = 2
                        total_servers_red = sum(initial_redundancy[j] * (len(comb[j]) * (h_add ** (len(comb[j]) - 1))) for j in range(len(comb)))
                        if total_servers_red <= H:   
                            software_availability = [calc_software_av(group, service_avail) * server_avail for group in comb]
                            system_avail_red = np.prod([1 - (1 - sa) ** int(r) for sa, r in zip(software_availability, initial_redundancy)])
                            redundancy_cost_efficiency.append((system_avail_red - system_avail) / (total_servers_red - total_servers))
                        else:
                            redundancy_cost_efficiency.append(0)
                        initial_redundancy[i] = 1  # Reset to 1
                    
                    avg_efficiency = np.mean(redundancy_cost_efficiency)
                    p_comb.append(str(comb))
                    p_rue.append(avg_efficiency)
            all_red.append(sw_redundancies)

        placement_result_dict = dict(zip(p_comb, p_rue))
        p_max_comb = []
        p_max_RCE = []

        # Sort combinations by RCE
        for i in range(len(p_comb)):
            max_pk, max_pv = max(placement_result_dict.items(), key=lambda x: x[1])
            p_max_comb.append(ast.literal_eval(max_pk))
            p_max_RCE.append(max_pv)
            del placement_result_dict[max_pk]

        unav, red, comb, before_red_RCE = search_best_redundancy(p_max_comb, all_red, p_max_RCE)

        end = time.time()
        time_diff = end - start  # Calculate elapsed time
        print(f"n={n}, h_add={h_add}, time={time_diff} seconds")

        # Save results to CSV
        output_filename = f"results_n={n}_h_add={h_add}.csv"
        save_results_to_csv(output_filename, p_comb, p_rue, unav, red, comb, before_red_RCE)
        print(f"Results have been saved to {output_filename}")

        # Normalize and plot
        RCE_sort = sorted(before_red_RCE)
        mm = preprocessing.MinMaxScaler()
        RCE = mm.fit_transform(np.array(before_red_RCE).reshape(-1, 1))
        ax.plot(RCE, unav, ".-", label=f"$\\mathrm{{r_{{add}}}}={h_add}, M={n}$")

ax.set_xlabel("RCE", fontsize=20)
ax.set_ylabel("非可用性", fontsize=20)
ax.set_yscale('log')
ax.legend(fontsize=14)

plt.savefig("RCE-Unavail-log.svg")
plt.savefig("RCE-Unavail-log.png")
plt.show()
