import time
import matplotlib.pyplot as plt
import cupy as cp  # NumPyの代わりにCupyをインポート
from itertools import combinations, chain, product
from numba import njit

# パラメータ
Resource = [30]  # サーバリソース
h_adds = [0.5,1,1.5]  # サービス数が1増えるごとに使うサーバ台数の増加

# 定数
num_service = [i for i in range(5, 15)]  # サービス数
server_avail = 0.99
max_redundancy = 5

# ソフトウェアの可用性を計算する関数
@njit
def calc_software_av(services_group, service_avail, services):
    indices = [services.index(s) for s in services_group]
    result = 1.0
    for i in indices:
        result *= service_avail[i]
    return result

def generate_service_combinations(services, num_software):
    all_combinations = []
    n = len(services)
    for indices in combinations(range(n - 1), num_software - 1):
        split_indices = list(chain([-1], indices, [n - 1]))
        combination = [services[split_indices[i] + 1: split_indices[i + 1] + 1] for i in range(len(split_indices) - 1)]
        all_combinations.append(combination)
    return all_combinations

# 冗長化の組み合わせを生成する関数
def generate_redundancy_combinations(num_software):
    return list(product(range(1, max_redundancy), repeat=num_software))

# システムの可用性を計算する関数
@njit
def calc_system_availability(software_availability, redundancy):
    system_avail = 1.0
    for sa, r in zip(software_availability, redundancy):
        system_avail *= (1 - (1 - sa) ** int(r))
    return system_avail

software_result = []
unav_list = []
time_list = []

for n in num_service:
    softwares = [i for i in range(1, n + 1)]
    services = [i for i in range(1, n + 1)]
    service_avail = [0.99] * n
    unav_list = []
    time_list = []
    
    for r_add in h_adds:
        for H in Resource:
            start = time.time()
            placement_result = []
            answer = []
            
            for num_software in softwares:
                all_combinations = generate_service_combinations(services, num_software)
                all_redundancies = generate_redundancy_combinations(num_software)

                # サービス実装形態によるCDFの計算
                p_results = []
                for comb in all_combinations:

                    max_system_avail = 0
                    best_redundancy = None

                    # software_availability の計算をループ外に移動
                    software_availability = cp.array([calc_software_av(group, service_avail, services) * server_avail for group in comb])
                    sw_resource = cp.array([r_add * (len(group) - 1) + 1 for group in comb])

                    # ここから最適な冗長化の探索を行う
                    for redundancy in all_redundancies:
                        red_array = cp.array(redundancy)
                        sw_red_resource = sw_resource * red_array
                        total_servers = cp.sum(sw_red_resource)
                        
                        if total_servers <= H:
                            # 最適化されたsystem_avail計算
                            system_avail = calc_system_availability(software_availability, redundancy)
                            if system_avail > max_system_avail:
                                max_system_avail = system_avail
                                best_redundancy = redundancy
                    
                    if best_redundancy is not None:
                        p_results.append((comb, best_redundancy, max_system_avail))
                        max_avails = [max_avail for _, _, max_avail in p_results]
                        max_idx = max_avails.index(max(max_avails))
                        answer.append(p_results[max_idx])
                        placement_result.append(max(max_avails))
                    else:
                        continue
                        
                a = max(placement_result)
                a_idx = placement_result.index(a)
                #print(answer[a_idx])

            end = time.time()
            time_diff = end - start
            time_list.append(time_diff)
            unav_list.append(1 - max(placement_result))

    print(f"{n}-result")
    for i in range(len(h_adds) * len(Resource)):
        print(time_list[i])
    for i in range(len(h_adds) * len(Resource)):
        print(unav_list[i])