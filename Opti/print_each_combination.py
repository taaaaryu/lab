import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import numpy as np
from itertools import combinations, chain, product
from matplotlib.colors import to_rgba

# パラメータ
Resourse = [15,20,25]  # サーバリソース
h_adds = [0.5,1,1.5]  # サービス数が1増えるごとに使うサーバ台数の増加


# 定数
n = 10  # サービス数
softwares = [i for i in range(1, n+1)]
services = [i for i in range(1, n + 1)]
service_avail = [0.99]*n
#service_avail = [0.9, 0.99, 0.99, 0.99, 0.99, 0.9, 0.99, 0.99, 0.99, 0.99]
server_avail = 0.99
max_redundancy = 5

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


# プロットを作成

plt.figure(figsize=(12, 8))

for H in Resourse:
    alloc = H*0.95  #サーバリソースの下限

    for h_add in h_adds:
        all_combinations_result = []
        placement_result = []
        ok_software = []
        
        for num_software in softwares:
            all_combinations = generate_service_combinations(services, num_software)
            all_redundancies = generate_redundancy_combinations(num_software, H, h_add)
            #progress_tqdm = tqdm(total = len(all_combinations)+len(all_redundancies), unit = "count")
            
            # サービス実装形態によるCDFの計算
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
            
            if len(p_results)!=0:
                max_avails = [max_avail for _, _, max_avail in p_results]
                max_soft_placement = max(max_avails)
                placement_result.extend(max_avails)
                ok_software.append(num_software)
                all_combinations_result.append(len(max_avails))
                
        plt.plot(ok_software, all_combinations_result, label=f"H={H}, h_add={h_add}")


plt.title("number of combinations in service inplement" )
plt.xlabel("number of service")
plt.ylabel("number of combinations")
plt.show()
