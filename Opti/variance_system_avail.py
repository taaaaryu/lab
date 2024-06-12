import time
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations, chain
from matplotlib.colors import to_rgba

# 定数
n = 10  # サービス数
services = [i for i in range(1, n + 1)]
service_availabilities = [
    [0.7, 0.7, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
    [0.7, 0.9, 0.9, 0.9, 0.9, 0.7, 0.9, 0.9, 0.9, 0.9],
    [0.7, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.7]
]
server_avail = 0.9
h_add = 0.5  # サービス数が1増えるごとに使うサーバ台数の増加
H_values = range(20, 31)  # サーバの台数範囲

start_time = time.time()  # 処理開始時間

# ソフトウェアの可用性を計算する関数
def calc_software_av(services_group, service_avail):
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

# 複数のサービス可用性配列に対する分散を計算し、プロット
fig, ax = plt.subplots(figsize=(10, 6))

for service_avail in service_availabilities:
    variances = []
    max_availabilities = []
    print(f"h_add = {h_add}, service_avail = {service_avail}")
    for H in H_values:
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
                software_availability = [calc_software_av(group, service_avail) * server_avail for group in comb]
                system_avail = np.prod([1 - (1 - sa) ** int(r) for sa, r in zip(software_availability, num_r)])
                system_unavailabilities.append(1 - system_avail)
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

        # システム可用性の分散を計算
        system_availability = [1 - unavail for unavail in system_unavailabilities]
        availability_variance = np.var(system_availability)
        variances.append(availability_variance)
        max_availabilities.append(max(system_availability))

    # 分散をプロット
    ax.plot(H_values, variances, linestyle="solid", marker='o', label=f"service_avail = {service_avail}")

    # 最大システム可用性をアノテーションとして追加
    for i, H in enumerate(H_values):
        ax.annotate(f'{max_availabilities[i]:.4f}', (H, variances[i]), textcoords="offset points", xytext=(0,10), ha='center')

# ラベルを追加
ax.set_xlabel("Server Resources (H)")
ax.set_ylabel("Variance of System Availability")
ax.set_title(f"Variance of System Availability  n = {10}, server_avail = {server_avail}, h_add = {h_add}")
ax.legend(loc="best")

end_time = time.time()  # 処理終了時間
elapsed_time = end_time - start_time  # 経過時間
print("Elapsed time: {:.2f} seconds".format(elapsed_time))

plt.grid(True)
plt.show()
