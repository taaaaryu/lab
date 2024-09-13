import time
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations, chain, product
import multiprocessing as mp

# パラメータ
Resource = [15]  # サーバリソース
h_adds = [1.5]  # サービス数が1増えるごとに使うサーバ台数の増加

# 定数
n = 10  # サービス数
softwares = [i for i in range(1, n + 1)]
services = [i for i in range(1, n + 1)]
service_avail = [0.99] * n
server_avail = 0.99
max_redundancy = 5

# ソフトウェアの可用性を計算する関数
def calc_software_av(services_group, service_avail, services):
    indices = [services.index(s) for s in services_group]
    result = 1.0
    for i in indices:
        result *= service_avail[i]
    return result

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
def generate_redundancy_combinations(num_software):
    return product(range(1, max_redundancy), repeat=num_software)

# 冗長化と可用性の計算を行う関数
def calculate_redundancy_availability(args):
    redundancy, sw_resource, H, alloc, software_availability = args
    red_array = np.array(redundancy)
    print(sw_resource)
    sw_red_resource = sw_resource * red_array
    total_servers = np.sum(sw_red_resource)
    if total_servers <= H and alloc <= total_servers:
        system_avail = np.prod([1 - (1 - sa) ** int(r) for sa, r in zip(software_availability, redundancy)])
        return system_avail, redundancy
    return None


def main():
    software_result = []
    unav_list = []
    time_list = []

    for r_add in h_adds:
        for H in Resource:
            alloc = H * 0.95  # サーバリソースの下限
            start = time.time()
            placement_result = []

            for num_software in softwares:
                all_combinations = generate_service_combinations(services, num_software)
                all_redundancies = generate_redundancy_combinations(num_software)

                # サービス実装形態によるCDFの計算
                p_results = []
                for comb in all_combinations:
                    
                    print(comb)
                    num_software = len(comb)

                    max_system_avail = -1
                    best_redundancy = None

                    # software_availability の計算をループ外に移動
                    software_availability = [calc_software_av(group, service_avail, services) * server_avail for group in comb]
                    sw_resource = np.array([r_add * (len(group) - 1) + 1 for group in comb])

                    # 並列化された冗長化の探索
                    with mp.Pool(mp.cpu_count()) as pool:
                        args_list = [(redundancy, sw_resource, H, alloc, software_availability) for redundancy in all_redundancies]
                        results = pool.map(calculate_redundancy_availability, args_list)

                    # 結果の処理
                    for result in results:
                        if result:
                            system_avail, redundancy = result
                            if system_avail > max_system_avail:
                                max_system_avail = system_avail
                                best_redundancy = redundancy

                    if best_redundancy:
                        p_results.append((comb, best_redundancy, max_system_avail))
                        max_avails = [max_avail for _, _, max_avail in p_results]
                        placement_result.append(max(max_avails))

            end = time.time()
            time_diff = end - start
            print(f"time = {time_diff}")

            time_list.append(time_diff)
            unav_list.append(1 - max(placement_result))

    # 結果の表示
    for i in range(len(h_adds) * len(Resource)):
        print(time_list[i])
    for i in range(len(h_adds) * len(Resource)):
        print(unav_list[i])

# メインブロック
if __name__ == '__main__':
    main()
