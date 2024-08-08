import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import numpy as np
from itertools import combinations, chain, product
from matplotlib.colors import to_rgba

# パラメータ
H = 20  # サーバリソース
h_add= 1  # サービス数が1増えるごとに使うサーバ台数の増加
POP = 20  #上位いくつの組み合わせを見るか

# 定数
n = 10  # サービス数
softwares = [i for i in range(5, n+1)]
services = [i for i in range(1, n + 1)]
service_avail = [0.99]*n
#service_avail = [0.9, 0.99, 0.99, 0.99, 0.99, 0.9, 0.99, 0.99, 0.99, 0.99]
server_avail = 0.99
alloc = H*0.9  #サーバリソースの下限
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
fig, ax = plt.subplots(2,1,figsize=(12, 8))

p_comb = []
p_av = []
r_comb = []
r_av = []

for num_software in softwares:
    all_combinations = generate_service_combinations(services, num_software)
    all_redundancies = generate_redundancy_combinations(num_software, H, h_add)
    progress_tqdm = tqdm(total = len(all_combinations)+len(all_redundancies), unit = "count")

    # サービス実装形態によるCDFの計算  
    """for comb in all_combinations:
        system_avail = 1
        sw_avail = [(calc_software_av(group,service_avail)**2)*server_avail for group in comb]
        for i in sw_avail:
            system_avail *= i
        p_comb.append(str(comb))
        p_av.append(system_avail)
        progress_tqdm.update(1)"""

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
            p_comb.append(str(comb))
            p_av.append(max_system_avail)
        progress_tqdm.update(1)

    # 冗長化度合いによるCDFの計算
    results = []
    for redundancy in all_redundancies:
        max_system_avail = -1
        best_combination = None
        for comb in all_combinations:
            total_servers = sum(redundancy[i] * ((h_add*(len(comb[i])-1))+1) for i in range(len(comb)))
            if total_servers <= H:
                if alloc <= total_servers:
                    software_availability = [calc_software_av(group, service_avail) * server_avail for group in comb]
                    system_avail = np.prod([1 - (1 - sa) ** int(r) for sa, r in zip(software_availability, redundancy)])
                    if system_avail > max_system_avail:
                        max_system_avail = system_avail
                        best_combination = comb
        if best_combination:
            r_comb.append(str(best_combination))
            r_av.append(max_system_avail)
        progress_tqdm.update(1)

#ax.plot(software_sx, software_sy, label=label3,color = "b")
progress_tqdm.close()

placement_result = dict(zip(p_comb,p_av))
redundancy_result = dict(zip(r_comb,r_av))

same = 0
p_max_comb = []
r_max_comb = []

for i in range(POP):
    max_pk, max_pv = max(placement_result.items(), key=lambda x: x[1])
    max_rk, max_rv = max(redundancy_result.items(), key=lambda x: x[1])
    p_max_comb.append(max_pk)
    r_max_comb.append(max_rk)
    print(f"placement{max_pk}, redundancy{max_rk} max_av{max_rv}")
    del placement_result[max_pk]
    del redundancy_result[max_rk]

for p in p_max_comb:
    if p in r_max_comb:
        same += 1
print(same)





