import csv
import time
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations, chain, product
from numba import njit

# パラメータ
h_adds = [0.8, 1, 1.2]  # サービス数が1増えるごとに使うサーバ台数の増加
num_service = [i for i in range(6, 14)]  # サービス数
server_avail = 0.995  # サーバの可用性
max_redundancy = 5
num_repeat = 10  # 各条件での計算を繰り返す回数

# CSV保存関数
def save_results_to_csv(file_path, h_adds, num_service, avg_time_list, max_time_list, min_time_list, 
                        avg_unav_list, max_unav_list, min_unav_list):
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # ヘッダーを書き込む
        writer.writerow(['r_add', 'num_service', 'time_avg', 'time_max', 'time_min', 'unav_avg', 'unav_max', 'unav_min'])

        # データを書き込む
        index = 0
        for ns in num_service:
            for h_add in h_adds:
                writer.writerow([
                    h_add,ns,
                    avg_time_list[index], max_time_list[index], min_time_list[index],
                    avg_unav_list[index], max_unav_list[index], min_unav_list[index]
                ])
                index += 1

    print(f"CSVファイル '{file_path}' が作成されました。")

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

# 結果を保存するリスト
avg_time_list = []
max_time_list = []
min_time_list = []
avg_unav_list = []
max_unav_list = []
min_unav_list = []

for n in num_service:
    softwares = [i for i in range(1, n + 1)]
    services = [i for i in range(1, n + 1)]
    service_avail = [0.999] * n
    Resource = [n * 2]  # サーバリソース

    for h_add in h_adds:
        for H in Resource:
            # リストを初期化
            time_values = []
            unav_values = []

            for _ in range(num_repeat):
                start = time.time()
                placement_result = []

                for num_software in softwares:
                    all_combinations = generate_service_combinations(services, num_software)
                    all_redundancies = generate_redundancy_combinations(num_software)

                    for comb in all_combinations:
                        max_system_avail = 0
                        best_redundancy = None

                        software_availability = np.array([
                            calc_software_av(group, service_avail, services) * server_avail
                            for group in comb
                        ])
                        sw_resource = np.array([
                            len(group) * (h_add ** (len(group) - 1)) for group in comb
                        ])

                        for redundancy in all_redundancies:
                            red_array = np.array(redundancy)
                            sw_red_resource = sw_resource * red_array
                            total_servers = np.sum(sw_red_resource)

                            if total_servers <= H:
                                system_avail = calc_system_availability(software_availability, redundancy)
                                if system_avail > max_system_avail:
                                    max_system_avail = system_avail
                                    best_redundancy = redundancy

                        if best_redundancy is not None:
                            placement_result.append(max_system_avail)

                end = time.time()
                time_diff = end - start
                time_values.append(time_diff)
                unav_values.append(1 - max(placement_result, default=0))

            # 平均、最大、最小を計算してリストに保存
            avg_time_list.append(np.mean(time_values))
            max_time_list.append(np.max(time_values))
            min_time_list.append(np.min(time_values))

            avg_unav_list.append(np.mean(unav_values))
            max_unav_list.append(np.max(unav_values))
            min_unav_list.append(np.min(unav_values))

# CSVに保存
save_results_to_csv(
    'results_全探索.csv',
    h_adds, num_service,
    avg_time_list, max_time_list, min_time_list,
    avg_unav_list, max_unav_list, min_unav_list
)
