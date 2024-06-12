import time
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations, chain
from matplotlib.colors import to_rgba

# 定数
n = 10  # サービス数
services = [i for i in range(1, n + 1)]
#service_avail = [0.7, 0.9, 0.9, 0.9, 0.9, 0.7, 0.9, 0.9, 0.9, 0.9]
service_avail = [0.99]*n
server_avail = 0.99
h_add = 1  # サービス数が1増えるごとに使うサーバ台数の増加
server_resources = [25,26,27,28,29,30]  # 複数のサーバリソースを設定

start_time = time.time()  # 処理開始時間

# ソフトウェアの可用性を計算する関数
def calc_software_av(services_group):
    indices = [services.index(s) for s in services_group]
    return np.prod([service_avail[i] for i in indices])

# サービスの組み合わせを生成する関数
def generate_service_combinations(services):
    all_combinations = []
    n = len(services)
    for num_software in range(1, n + 1):
        for indices in combinations(range(n - 1), num_software - 1):
            split_indices = list(chain([-1], indices, [n - 1]))
            combination = [services[split_indices[i] + 1: split_indices[i + 1] + 1] for i in range(len(split_indices) - 1)]
            all_combinations.append(combination)
    return all_combinations

# サービスのすべての組み合わせを生成
all_combinations = generate_service_combinations(services)

# 複数のサーバリソースに対するプロットを作成
fig, ax = plt.subplots(figsize=(10, 6))

for H in server_resources:
    system_unavailabilities = []
    configurations = []
    redundancies = []

    for comb in all_combinations:
        redundancy = [1] * len(comb)  # 冗長化してないときのソフトウェアごとに使用するサーバリソースを計算
        num_r = [1] * len(comb)
        for i in range(len(comb)):
            redundancy[i] += (len(comb[i]) - 1) * h_add
        total_servers = sum(redundancy)
        base_redundancy = redundancy.copy()

        while total_servers <= H:
            software_availability = [calc_software_av(group) * server_avail for group in comb]
            system_avail = np.prod([1 - (1 - sa) ** int(r) for sa, r in zip(software_availability, num_r)])
            system_unavailabilities.append(1-system_avail)
            configurations.append(comb)
            redundancies.append(num_r.copy())

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
                new_software_availability = [calc_software_av(group) * server_avail for group in comb]
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

    # 上位5つのシステム可用性の点を見つける
    sorted_indices = np.argsort(system_unavailabilities)[-5:]

    # 最大の可用性を求める
    max_availability = max(1 - unavail for unavail in system_unavailabilities)
    max_availability_index = system_unavailabilities.index(1 - max_availability)

    # システム可用性の分散を計算
    system_availability = [1 - unavail for unavail in system_unavailabilities]
    availability_variance = np.var(system_availability)

    # システム非可用性をプロット
    density_reduction_factor = 1  # プロット密度を減らすための間隔
    ax.plot(system_unavailabilities[::density_reduction_factor], linestyle="solid", marker='o', label=f"H = {H}")

    # 上位5つのシステム可用性の点にアノテーションを追加
    for idx in sorted_indices:
        print(f'Config: {configurations[idx]}\nRedundancy: {redundancies[idx]}',(idx // density_reduction_factor, system_unavailabilities[idx]))
        """ax.annotate(f'Config: {configurations[idx]}\nRedundancy: {redundancies[idx]}',
                    (idx // density_reduction_factor, system_unavailabilities[idx]), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=8, color='red')"""

    # 最大の可用性の点にアノテーションを追加
    ax.annotate(f'Max Availability\nConfig: {configurations[max_availability_index]}\nRedundancy: {redundancies[max_availability_index]}',
                (max_availability_index // density_reduction_factor, system_unavailabilities[max_availability_index]), 
                textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='green')

    # システム可用性の分散を表示
    ax.text(0.05, 0.95 - 0.1 * server_resources.index(H), f"Variance (H={H}): {availability_variance:.6f}", transform=ax.transAxes, fontsize=10, verticalalignment='top')

# ラベルを追加
ax.set_xlabel(f"all service combination (H={H})")
ax.set_ylabel("System Unavailability")
ax.set_title(f"System Unavailability for Different Configurations (n = {n}, h_add = {h_add})")
ax.set_yscale('log')
ax.legend(loc="best")
end_time = time.time()  # 処理終了時間
elapsed_time = end_time - start_time  # 経過時間
print("Elapsed time: {:.2f} seconds".format(elapsed_time))

plt.grid(True)
plt.show()


