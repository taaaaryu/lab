import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import numpy as np
from itertools import combinations, chain, product
import matplotlib.cm as cm

# パラメータ
n = 10  # サービス数
H = n * 2  # サーバリソース
h_add_values = [0.5, 0.75, 1.0, 1.25, 1.5]  # サービス数が1増えるごとに使うサーバ台数の増加

# 定数
softwares = [i for i in range(1, n + 1)]
services = [i for i in range(1, n + 1)]
service_avail = [0.95] * n
server_avail = 0.99
alloc = H * 0.95  # サーバリソースの下限
max_redundancy = 5
top_x_percent = 0.01  # 上位x%の設定

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

# メインの処理
results = []
counts = []

colors = cm.winter(np.linspace(0, 1, len(h_add_values)))

for h_add, color in zip(h_add_values, colors):
    software_availabilities = []
    combination_lengths = []
    for num_software in softwares:
        all_combinations = generate_service_combinations(services, num_software)
        all_redundancies = generate_redundancy_combinations(num_software, H, h_add)
        progress_tqdm = tqdm(total=len(all_combinations) + len(all_redundancies), unit="count")

        redundancy_result = []

        # 冗長化度合いによるCDFの計算
        for redundancy in all_redundancies:
            max_system_avail = -1
            best_combination = None
            for comb in all_combinations:
                total_servers = sum(redundancy[i] * ((h_add * (len(comb[i]) - 1)) + 1) for i in range(len(comb)))
                if total_servers <= H:
                    if alloc <= total_servers:
                        software_availability = [calc_software_av(group, service_avail) * server_avail for group in comb]
                        system_avail = np.prod([1 - (1 - sa) ** int(r) for sa, r in zip(software_availability, redundancy)])
                        if system_avail > max_system_avail:
                            max_system_avail = system_avail
                            best_combination = comb
            if best_combination:
                results.append((redundancy, best_combination, max_system_avail, h_add))
                combination_lengths.append((max_system_avail, len(best_combination), h_add))
            progress_tqdm.update(1)

        progress_tqdm.close()
    # 上位x%のソフトウェア数を計算
    combination_lengths.sort(reverse=True, key=lambda x: x[0])
    top_x_count = int(top_x_percent * len(combination_lengths))

    # 上位x%のシステム可用性に対応するbest_combinationの長さの分布
    top_combination_lengths = [length for avail, length, h_add in combination_lengths[:top_x_count]]
    count_software = [top_combination_lengths.count(i) for i in range(1, n + 1)]
    counts.append((h_add, count_software))


# ソフトウェア数、h_add、カウントを記録
software_count = np.arange(1, n + 1)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

for (h_add, count_software), color in zip(counts, colors):
    ax.plot(software_count, [h_add] * n, count_software, marker='o', label=h_add,color=color, linewidth=1)

ax.set_xlabel('Number of Software')
ax.set_ylabel('r_add')
ax.set_zlabel('Counts')
ax.set_title(f'Top {top_x_percent * 100}% Availability')
ax.legend()

plt.show()
