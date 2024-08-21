import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import numpy as np
from itertools import combinations, chain, product
from matplotlib.colors import to_rgba

# パラメータ
Resourse = [20]  # サーバリソース
h_adds = [1]  # サービス数が1増えるごとに使うサーバ台数の増加


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


for H in Resourse:
    alloc = H*0.95  #サーバリソースの下限

    for h_add in h_adds:
        software_result = []
        placement_result = []
        redundancy_result = []
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for num_software in softwares:
            all_combinations = generate_service_combinations(services, num_software)
            all_redundancies = generate_redundancy_combinations(num_software, H, h_add)
            progress_tqdm = tqdm(total = len(all_combinations)+len(all_redundancies), unit = "count")
            

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
                progress_tqdm.update(1)
            
            if len(p_results)!=0:
                max_unavails = [1-max_avail for _, _, max_avail in p_results]
                max_soft_placement = min(max_unavails)
                placement_result.extend(max_unavails)


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
                    results.append((redundancy, best_combination, max_system_avail))
                progress_tqdm.update(1)

            # カバーされている冗長化組み合わせを削除
            '''optimized_results = []
            for redundancy, comb, max_avail in results:
                if not any(all(r_old >= r_new for r_old, r_new in zip(existing[0], redundancy)) for existing in results if existing[0] != redundancy):
                    optimized_results.append((redundancy, comb, max_avail))'''
            
            if len(results)!=0:
                max_unavails = [1-max_avail for _, _, max_avail in results]
                max_soft_redundancy = min(max_unavails)
                redundancy_result.extend(max_unavails)
                software_result.append([num_software,max_soft_redundancy])

        #ax.plot(software_sx, software_sy, label=label3,color = "b")


        label1 = f"Service_Implementation"
        label2 = f"Redundancy"
        label3 = f"Software"

        placement_sx = sorted(placement_result)
        placement_sx.reverse()
        print(1-min(placement_sx))
        N = len(placement_sx)
        placement_sy = [i / (N-1) for i in range(N)]
        print(max(placement_result))

        redundancy_sx = sorted(redundancy_result)
        redundancy_sx.reverse()
        N = len(redundancy_sx)
        redundancy_sy = [i / (N-1) for i in range(N)]
        print(max(redundancy_result))

        software_sx = sorted(software_result, key=lambda x: x[1])
        software_sx.reverse()
        N = len(software_sx)
        software_sy = [i / (N-1) for i in range(N)]


        ax.plot(placement_sx, placement_sy, label=label1, color="blue" )
        ax.plot(redundancy_sx, redundancy_sy, label=label2, color = "red")
        ax.plot([x[1] for x in software_sx], software_sy, label=label3, color="black")
        
        #ソフトウェアの結果をプロットする部分を修正
        """for x, y, num in zip([x[1] for x in software_sx], software_sy, [x[0] for x in software_sx]):
            ax.text(x, y, str(num), fontsize=20, ha='left', va='bottom')"""

        progress_tqdm.close()

        ax.set_xlabel('System Unavailability')
        ax.set_ylabel('CDF')
        ax.set_xscale('log')
        #ax.set_xlim(0, 0.3)
        ax.legend()
        ax.set_title(f"Service = {n}, Resource = {H}, r_add = {h_add}")


        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plt.savefig(f"{h_add}-{H}.png")
