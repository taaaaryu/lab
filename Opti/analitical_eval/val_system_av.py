import time
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations, chain, product
import ast

# Parameters
Resourse = [30]  # Server resource
h_adds = [0.75]  # Increment in server count per additional service
POP = 0.02  # Top combinations to consider

# Constants
n = 10  # Number of services
softwares = [i for i in range(1, n+1)]
services = [i for i in range(1, n + 1)]
service_avail = [0.99]*n
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
def search_best_redundancy(all_combinations, all_redundancies):
    r_av = []
    best_reds = []
    for comb in all_combinations:
        each_redundancy = all_redundancies[len(comb) - 1]
        max_system_avail = -1
        best_redundancy = None
        for redundancy in each_redundancy:
            total_servers = sum(redundancy[i] * ((h_add*(len(comb[i])-1))+1) for i in range(len(comb)))
            if total_servers <= H:
                if alloc <= total_servers:
                    software_availability = [calc_software_av(group, service_avail) * server_avail for group in comb]
                    system_avail = np.prod([1 - (1 - sa) ** int(r) for sa, r in zip(software_availability, redundancy)])
                    if system_avail > max_system_avail:
                        max_system_avail = system_avail
                        best_redundancy = redundancy
        if best_redundancy:
            r_av.append(max_system_avail)
            best_reds.append(best_redundancy)
    return r_av, best_reds

# Main process
for H in Resourse:
    alloc = H*0.95  # Minimum server resource allocation

    for h_add in h_adds:
        software_result = []
        placement_result = []
        redundancy_result = []
        fig, ax = plt.subplots(figsize=(12, 8))

        for num_software in softwares:
            all_combinations = generate_service_combinations(services, num_software)
            all_redundancies = generate_redundancy_combinations(num_software, H, h_add)

            # Calculate CDF for service implementation
            p_results = []
            for comb in all_combinations:
                max_system_avail = -1
                best_redundancy = None
                for redundancy in all_redundancies:
                    total_servers = sum(redundancy[i] * ((h_add*(len(comb[i])-1))+1) for i in range(len(comb)))
                    if total_servers <= H:
                        if alloc <= total_servers:
                            software_availability = [calc_software_av(group, service_avail) * server_avail for group in comb]
                            system_avail = np.prod([1 - (1 - sa) ** int(r) for sa, r in zip(software_availability, redundancy)])
                            if system_avail > max_system_avail:
                                max_system_avail = system_avail
                                best_redundancy = redundancy
                if best_redundancy:
                    p_results.append((comb, best_redundancy, max_system_avail))

            if len(p_results) != 0:
                max_unavails = [1-max_avail for _, _, max_avail in p_results]
                max_soft_placement = min(max_unavails)
                placement_result.extend(max_unavails)

        # Plot CDF for service implementation
        placement_sx = sorted(placement_result)
        placement_sx.reverse()
        N = len(placement_sx)
        placement_sy = [i / (N-1) for i in range(N)]
        ax.plot(placement_sx, placement_sy, label="Service Implementation", color="blue")

        # Calculate and plot CDF for system availability after redundancy
        p_comb = []
        p_rue = []
        all_red = []
        for num_software in softwares:
            all_combinations = generate_service_combinations(services, num_software)
            sw_redundancies = generate_redundancy_combinations(num_software, H, h_add)
            for comb in all_combinations:
                max_system_avail = None
                total_servers = sum((h_add*(len(comb[i])-1)+1) for i in range(len(comb)))
                if total_servers <= H:
                    software_availability = [calc_software_av(group, service_avail) * server_avail for group in comb]
                    system_avail = np.prod([sa for sa in software_availability])
                    
                    # Set initial redundancy to 1 for all and then increase for one software
                    initial_redundancy = [1] * len(comb)
                    redundancy_cost_efficiency = []
                    
                    for i in range(len(comb)):
                        initial_redundancy[i] = 2
                        total_servers_red = sum(initial_redundancy[j] * ((h_add*(len(comb[j])-1))+1) for j in range(len(comb)))
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
        #print(placement_result_dict)
        p_max_comb = []
        print(POP*len(p_comb))
        
        count_software = [0]*len(softwares)
        
        for i in range(int(POP*len(p_comb))):
            max_pk, max_pv = max(placement_result_dict.items(), key=lambda x: x[1])
            good_comb =  ast.literal_eval(max_pk)
            count_software[len(good_comb)-1] += 1
            p_max_comb.append(ast.literal_eval(good_comb))
            del placement_result_dict[max_pk]
        
        x = [i+1 for i in range(len(softwares))]
        
        ax.bar(x, count_software, edgecolor='black')
        ax.set_xlabel('Number of Software')
        ax.set_ylabel('Counts')
        ax.set_title(f'Top {top_x_count}% Availability')
        ax.set_xticks(x)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


        """
        av, red = search_best_redundancy(p_max_comb, all_red)

        # Add `av` values to the CDF plot
        av_sorted = sorted([1-a for a in av],reverse=True)
        av_sy = [i / (len(av_sorted) - 1) for i in range(len(av_sorted))]
        ax.plot(av_sorted, av_sy, label="System Availability TOP10", color="green", linestyle="--")

        ax.set_title(f'H = {H}, h_add = {h_add}', fontsize=14)
        ax.set_xlabel("Unavailability", fontsize=12)
        ax.set_ylabel("CDF", fontsize=12)
        ax.legend()
        #plt.show()
        plt.savefig(f"val_{h_add}-{H}.png")
        print(h_add)
        """