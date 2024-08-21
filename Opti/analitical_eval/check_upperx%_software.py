import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import numpy as np
from itertools import combinations, chain, product
from matplotlib.colors import to_rgba
import statistics

# パラメータ
n = 10  # サービス数
resources = [15,20,25]  # サーバリソース
h_adds = [0.5,1,1.5]  # サービス数が1増えるごとに使うサーバ台数の増加

# 定数

softwares = [i for i in range(1, n+1)]
services = [i for i in range(1, n + 1)]
service_avail = [0.99]*n
#service_avail = [0.99,0.99,0.99,0.9,0.99,0.99,0.9,0.99,0.99,0.99]
server_avail = 0.99

max_redundancy = 5
top_x_count = 20  

# ソフトウェアの可用性を計算する関数
def calc_software_av(services_group, service_avail):
    indices = [services.index(s) for s in services_group]
    return np.prod([service_avail[i] for i in indices])

# サービスの組み合わせを生成する関数
def generate_service_combinations(services, num_software):
    all_combinations = []
    n = len(services)
    for indices in combinations(range(n - 1), num_software - 1):
        split_indices = list(chain([-1], indices, [n - 1]))
        combination = [services[split_indices[i] + 1: split_indices[i + 1] + 1] for i in range(len(split_indices) - 1)]
        all_combinations.append(combination)
    return all_combinations

# 冗長化の組み合わせを生成する関数
def generate_redundancy_combinations(num_software, max_servers, h_add):
    all_redundancies = [redundancy for redundancy in product(range(1, max_redundancy), repeat=num_software)]
    return all_redundancies

def calc_sw_resource(comb):
    sw = [(h_add * (len(i) - 1)) + 1 for i in comb]
    return sw

software_availabilities = []
combination_lengths = []
results = []
fig, ax = plt.subplots(figsize=(12, 8))
i=0
for H in resources:
    alloc = H * 0.95  # サーバリソースの下限
    for h_add in h_adds:
        software_availabilities = []
        combination_lengths = []
        results = []
        fig, ax = plt.subplots(figsize=(12, 8))
        i=0
        for num_software in softwares:
            all_combinations = generate_service_combinations(services, num_software)
            all_redundancies = generate_redundancy_combinations(num_software, H, h_add)
            
            placement_result = []

            # Calculate CDF for service implementation
            for comb in all_combinations:
                max_system_avail = -1
                best_redundancy = None
                for redundancy in all_redundancies:
                    total_servers = sum(redundancy[i] * ((h_add * (len(comb[i]) - 1)) + 1) for i in range(len(comb)))
                    if total_servers <= H:
                        if alloc <= total_servers:
                            software_availability = [calc_software_av(group, service_avail) * server_avail for group in comb]
                            system_avail = np.prod([1 - (1 - sa) ** int(r) for sa, r in zip(software_availability, redundancy)])
                            if system_avail > max_system_avail:
                                max_system_avail = system_avail
                                best_redundancy = redundancy
                if best_redundancy:
                    results.append((best_redundancy, comb, max_system_avail))
                    combination_lengths.append((max_system_avail, len(comb)))


        # 上位x%のシステム可用性に対応するbest_combinationの長さの分布
        results.sort(key=lambda x: x[2], reverse=True)
        print(h_add)
        print(results[:20])
        combination_lengths.sort(key=lambda x: x[0], reverse=True)
        std_sw_resource = []
        std_sw_resource_all = []

        #全ての組み合わせの標準偏差
        comb_all = [comb for red, comb, avail in results]
        each_sw_resource_all = [calc_sw_resource(k) for k in comb_all]
        for j in each_sw_resource_all:
            if len(j)==1:
                continue
            std_sw_resource_all.append(statistics.stdev(j))
        
        top_combination_lengths = [length for avail, length in combination_lengths[:top_x_count]]
        
        
        #print(results[:20])
        #TOP1%の組み合わせのサービス数の標準偏差
        comb = [comb for red, comb, avail in results[:top_x_count]]
        each_service = [calc_sw_resource(k) for k in comb]
        for j in each_service:
            if len(j)==1:
                continue
            std_sw_resource.append(statistics.stdev(j))
        top_combination_lengths = [length for avail, length in combination_lengths[:top_x_count]]
        #print(top_combination_lengths)

        count_software = [top_combination_lengths.count(i) for i in range(1,num_software+1)]

        # 標準偏差のCDFを計算し、プロット
        std_sw_resource.sort()
        std_sw_resource_all.sort()
        cdf = np.arange(len(std_sw_resource)) / float(len(std_sw_resource) - 1)
        cdf_all = np.arange(len(std_sw_resource_all)) / float(len(std_sw_resource_all) - 1)
        
        ax.plot(std_sw_resource, cdf, marker='o', linestyle='-.',label = h_add, color=cm.jet(i/len(h_adds)))
        print(len(cdf), len(cdf_all))
        ax.plot(std_sw_resource_all, cdf_all, marker='x', linestyle = '-',label = "all_{h_add}", color=cm.jet(i/len(h_adds)))

        ax.set_xlabel('Standard Deviation of SW Resource')
        ax.set_ylabel('CDF')
        ax.legend()
        ax.set_title('CDF of Standard Deviation of SW Resource')
        plt.grid(True)
        plt.savefig(f"upper_x_{h_add}-{H}.png")

        # ソフトウェア数の分布をプロット
        fig, ax = plt.subplots(figsize=(12, 8))
        x = [i for i in range(1,num_software+1)]

        ax.bar(x, count_software, edgecolor='black')
        ax.set_xlabel('Number of Software')
        ax.set_ylabel('Counts')
        ax.set_title(f'Top {top_x_count}% Availability')
        ax.set_xticks(x)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"upper_sw_{h_add}-{H}.png")
