#全てのRCEを計算したのちに、最適な冗長化を行い、冗長化前のRCEと冗長化後の可用性の関係を探る

import time
import matplotlib.pyplot as plt
import math
import numpy as np
from itertools import combinations, chain, product
import ast
import japanize_matplotlib
import time

# Parameters
 # Server resource
h_adds = [0.8]  # Increment in server count per additional service
POP = 0.1  # Top combinations to consider

# Constants
n = 10  # Number of services
Resourse = [n*2] 
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
def search_best_redundancy(all_combinations, all_redundancies,RCE):
    r_unav = []
    best_reds = []
    best_comb = []
    exist_RCE=[] #RCEの高さとavailabilityの相関比較用にリスト作成
    for k in range(len(all_combinations)):
        comb = all_combinations[k]
        each_redundancy = all_redundancies[len(comb) - 1]
        max_system_avail = -1
        best_redundancy = None
        for redundancy in each_redundancy:
            total_servers = sum(redundancy[i] * (len(comb[i])*(h_add**(len(comb[i])-1))) for i in range(len(comb)))
            #print(comb,total_servers)
            if total_servers <= H:
                if alloc <= total_servers:
                    software_availability = [calc_software_av(group, service_avail) * server_avail for group in comb]
                    system_avail = np.prod([ 1- (1 - sa) ** int(r) for sa, r in zip(software_availability, redundancy)])
                    if system_avail > max_system_avail:
                        max_system_avail = system_avail
                        best_redundancy = redundancy
        if best_redundancy:
            r_unav.append(1 - max_system_avail)
            best_reds.append(best_redundancy)
            best_comb.append(comb)
            exist_RCE.append(RCE[k])

    return r_unav, best_reds, best_comb, exist_RCE

# Main process
for H in Resourse:
    alloc = H*0.9  # Minimum server resource allocation

    for h_add in h_adds:
        
        start = time.time()
        
        fig, ax = plt.subplots(figsize=(12, 8))

        # Calculate and plot CDF for system availability after redundancy
        p_comb = []
        p_rue = []
        all_red = []
        for num_software in softwares:
            all_combinations = generate_service_combinations(services, num_software)
            sw_redundancies = generate_redundancy_combinations(num_software, H, h_add)
            for comb in all_combinations:
                max_system_avail = None
                total_servers = sum(len(comb[i])*(h_add ** (len(comb[i]) - 1)) for i in range(len(comb)))
                if total_servers <= H:
                    software_availability = [calc_software_av(group, service_avail) * server_avail for group in comb]
                    system_avail =(np.prod([sa for sa in software_availability]))
                    
                    # Set initial redundancy to 1 for all and then increase for one software
                    initial_redundancy = [1] * len(comb)
                    redundancy_cost_efficiency = []
                    
                    for i in range(len(comb)):
                        initial_redundancy[i] = 2
                        total_servers_red = sum(initial_redundancy[j] * (len(comb[j])*(h_add ** (len(comb[j]) - 1))) for j in range(len(comb)))
                        #print(total_servers_red)
                        if total_servers_red <= H:   
                            software_availability = [calc_software_av(group, service_avail) * server_avail for group in comb]
                            system_avail_red = (np.prod([1 - (1 - sa) ** int(r) for sa, r in zip(software_availability, initial_redundancy)]))
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
        p_max_RCE = []
        
        #RCEの高い順に組み合わせを並び替える
        for i in range(len(p_comb)):
            max_pk, max_pv = max(placement_result_dict.items(), key=lambda x: x[1])
            p_max_comb.append(ast.literal_eval(max_pk))
            p_max_RCE.append(max_pv)
            del placement_result_dict[max_pk]

        a = 10

        unav, red, comb, before_red_RCE = search_best_redundancy(p_max_comb, all_red, p_max_RCE)
        
        end = time.time()
        time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する
        print(f"{h_add}, {H}")
        print(f"time = {time_diff}")  # 処理にかかった時間データを使用
        
        RCE_sort = sorted(before_red_RCE)
        line = RCE_sort[-a]


        ax.plot(before_red_RCE, unav, label = "サービス実装形態")
        #plt.vlines(line,0,1, color='g', linestyles='dotted', label = f"upper10")
    
        # Add `av` values to the CDF plot
        #av_sorted = sorted([1-a for a in av],reverse=True)
        #av_sy = [i / (len(av_sorted) - 1) for i in range(len(av_sorted))]
        #ax.plot(av_sorted, av_sy, label="System Availability After Redundancy", color="green", linestyle="--")

        ax.set_title(f'r_add = {h_add}, service = {n}', fontsize=14)
        ax.set_xlabel("RCE", fontsize=12)
        ax.set_ylabel("非可用性", fontsize=12)
        ax.set_yscale('log')

        ax.legend()
        #fig.subplots_adjust(left=0, right=1, bottom=0, top=1) 
        #plt.show()
        plt.savefig(f"RCE-Unavail-log_{h_add}.png")
        #print(h_add)