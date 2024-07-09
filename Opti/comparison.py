import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from itertools import combinations, chain, product
from matplotlib.colors import to_rgba

# 定数
n = 10  # サービス数
softwares = [i for i in range(1, 3)]
services = [i for i in range(1, n + 1)]
service_avail = [0.99]*n
#service_avail = [0.9, 0.99, 0.99, 0.99, 0.99, 0.9, 0.99, 0.99, 0.99, 0.99]
server_avail = 0.99
H = 15.0  # サーバの台数
h_add= 0.5  # サービス数が1増えるごとに使うサーバ台数の増加

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
    all_redundancies = [redundancy for redundancy in product(range(1, 5), repeat=num_software)]
    return all_redundancies

# プロットを作成
fig, ax = plt.subplots(figsize=(12, 8))

placement_result = []
redundancy_result = []
software_result = []

for num_software in softwares:
    all_combinations = generate_service_combinations(services, num_software)
    all_redundancies = generate_redundancy_combinations(num_software, H, h_add)
    print(len(all_redundancies))
    progress_tqdm = tqdm(total = len(all_combinations)+len(all_redundancies), unit = "count")

    # サービス実装形態によるCDFの計算
    save_system_av = []

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
                if new_total_servers > H or any(r > 4 for r in new_num_r):  # 追加: 冗長化数の上限を4に設定
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

        save_system_av.append(max_avail)
        progress_tqdm.update(1)
    max_soft_placement = max(save_system_av)
    software_result.append(max_soft_placement)
    placement_result.extend(save_system_av)

    # 冗長化度合いによるCDFの計算
    results = []
    for redundancy in all_redundancies:
        max_system_avail = -1
        best_combination = None
        for comb in all_combinations:
            total_servers = sum(redundancy[i] * ((h_add*(len(comb[i])-1))+1) for i in range(len(comb)))
            if total_servers <= H:
                software_availability = [calc_software_av(group, service_avail) * server_avail for group in comb]
                system_avail = np.prod([1 - (1 - sa) ** int(r) for sa, r in zip(software_availability, redundancy)])
                if system_avail > max_system_avail:
                    max_system_avail = system_avail
                    best_combination = comb
        if best_combination:
            results.append((redundancy, best_combination, max_system_avail))
        progress_tqdm.update(1)

    # カバーされている冗長化組み合わせを削除
    optimized_results = []
    for redundancy, comb, max_avail in results:
        if not any(all(r_old >= r_new for r_old, r_new in zip(existing[0], redundancy)) for existing in results if existing[0] != redundancy):
            optimized_results.append((redundancy, comb, max_avail))
    print(optimized_results)
    max_avails = [max_avail for _, _, max_avail in optimized_results]
    max_soft_redundancy = max(max_avails)
    redundancy_result.extend(max_avails)

    print(max_soft_placement)
    print(max_soft_redundancy)

# ラベルを追加
placement_sx = sorted(placement_result)
N = len(placement_sx)
placement_sy = [i / N for i in range(N)]

redundancy_sx = sorted(redundancy_result)
N = len(redundancy_sx)
redundancy_sy = [i / N for i in range(N)]

software_sx = sorted(software_result)
N = len(software_sx)
software_sy = [i / N for i in range(N)]

# プロット
label1 = f"placement"
label2 = f"redundancy"
label3 = f"software"

ax.plot(placement_sx, placement_sy, label=label1)
ax.plot(redundancy_sx, redundancy_sy, label=label2)
ax.plot(software_sx, software_sy, label=label3)

progress_tqdm.close()

ax.set_xlabel('System Availability')
ax.set_ylabel('CDF')
ax.set_xlim(0.8, 1.0)
ax.legend()
ax.set_title(f"n = {n}, resource = {H}, h_add = {h_add}")

plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.show()
