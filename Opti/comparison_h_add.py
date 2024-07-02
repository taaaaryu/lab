import time
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations, chain, product
from matplotlib.colors import to_rgba

# 定数
n = 10  # サービス数
num_software = 7
services = [i for i in range(1, n + 1)]
service_avail = [0.9, 0.99, 0.99, 0.99, 0.99, 0.9, 0.99, 0.99, 0.99, 0.99]
server_avail = 0.99
H = 25  # サーバの台数

h_add_values = [0.5, 1, 1.5]  # サービス数が1増えるごとに使うサーバ台数の増加

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
    all_redundancies = []
    for redundancy in product(range(1, max_servers // 2), repeat=num_software):
        if sum(redundancy) * h_add <= max_servers:
            all_redundancies.append(redundancy)
    return all_redundancies

colors = {
    0.5: 'red',
    1: 'blue',
    1.5: 'green'
}

# プロットを作成
fig, ax = plt.subplots(figsize=(12, 8))

for h_add in h_add_values:
    all_combinations = generate_service_combinations(services, num_software)
    all_redundancies = generate_redundancy_combinations(num_software, H, h_add)

    # サービス実装形態によるCDFの計算
    comb_max_system_avail = []
    system_av_forCDF = []

    for comb in all_combinations:
        max_avail = -1
        redundancy = [1] * len(comb)  # 冗長化してないときのソフトウェアごとに使用するサーバリソースを計算
        num_r = [1] * len(comb)
        for i in range(len(comb)):
            redundancy[i] += (len(comb[i]) - 1) * h_add
        total_servers = sum(redundancy)
        base_redundancy = redundancy.copy()

        while total_servers <= H:
            software_availability = [calc_software_av(group, service_avail) * server_avail for group in comb]
            system_avail = np.prod([1 - (1 - sa) ** int(r) for sa, r in zip(software_availability, num_r)])

            if system_avail > max_avail:
                max_avail = system_avail

            # システム可用性が最も上昇するように1つのソフトウェアを冗長化
            improvements = []
            for i in range(len(comb)):  # あるサービス組み合わせにおいて、1つのソフトウェアを冗長化
                new_redundancy = redundancy.copy()
                new_num_r = num_r.copy()
                new_redundancy[i] += base_redundancy[i]
                new_num_r[i] += 1
                new_total_servers = sum(new_redundancy)
                if new_total_servers > H:
                    continue
                new_software_availability = [calc_software_av(group, service_avail) * server_avail for group in comb]
                new_system_avail = np.prod([1 - (1 - sa) ** int(r) for sa, r in zip(new_software_availability, new_num_r)])
                improvement = new_system_avail - system_avail
                improvements.append((improvement, i, new_redundancy, new_num_r))

            if not improvements:
                break

            # 最大の改善をもたらす冗長化を適用
            best_improvement = max(improvements)
            redundancy = best_improvement[2]
            num_r = best_improvement[3]
            total_servers = sum(redundancy)

        comb_max_system_avail.append((comb, max_avail, num_r))
        system_av_forCDF.append(max_avail)

    placement_sx = sorted(system_av_forCDF)
    N = len(system_av_forCDF)
    placement_sy = [i / N for i in range(N)]

    # 冗長化度合いによるCDFの計算
    results = []
    for redundancy in all_redundancies:
        max_system_avail = -1
        best_combination = None
        for comb in all_combinations:
            total_servers = sum(redundancy[i] * (h_add*(len(comb[i])-1)+1) for i in range(len(comb)))
            if total_servers <= H:
                software_availability = [calc_software_av(group, service_avail) * server_avail for group in comb]
                system_avail = np.prod([1 - (1 - sa) ** int(r) for sa, r in zip(software_availability, redundancy)])

                if system_avail > max_system_avail:
                    max_system_avail = system_avail
                    best_combination = comb
        if best_combination:
            results.append((redundancy, best_combination, max_system_avail))

    max_avails = [max_avail for _, _, max_avail in results]
    redundancy_sx = sorted(max_avails)
    N = len(redundancy_sx)
    redundancy_sy = [i / N for i in range(N)]

    # プロット
    label1 = f"placement (h_add={h_add})"
    label2 = f"redundancy (h_add={h_add})"

    ax.plot(placement_sx, placement_sy, color=to_rgba(colors[h_add], 0.7), alpha=0.7, label=label1)
    ax.plot(redundancy_sx, redundancy_sy, color=to_rgba(colors[h_add], 1.0), alpha=0.7, linestyle='dashed', label=label2)

# ラベルを追加
ax.set_xlabel('System Availability')
ax.set_ylabel('CDF')
ax.set_xlim(0.7, 1.0)
ax.legend()
ax.set_title(f"n = {n}, resource = {H}, num_software = {num_software}")

plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.show()
