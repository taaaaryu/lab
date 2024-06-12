import time
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations, chain

# 定数
n = 15# サービス数
services = [i for i in range(1, n + 1)]
service_avail_1 = [0.9]*n
service_avail_2 = [0.7, 0.9 ,0.9, 0.9, 0.9, 0.7, 0.9, 0.9, 0.9, 0.9, 0.7, 0.9, 0.9, 0.9, 0.9]
service_avail_3 = [0.7, 0.7 ,0.9, 0.9, 0.9, 0.7, 0.7, 0.9, 0.9, 0.9, 0.7, 0.7, 0.9, 0.9, 0.9]
server_avail = 0.9
h_add = 1  # サービス数が1増えるごとに使うサーバ台数の増加
#H = 30  # 全体のリソース(今回は見ない)
redundancy = 4  # 冗長化数

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
num_comb = len(all_combinations)

# 冗長度合いを計算し、システム非可用性を記録
def calculate_system_result(service_avail):
    system_result = []
    for comb in all_combinations:
        software_availability = [calc_software_av(group, service_avail) * server_avail for group in comb]
        system_unavail = 1 - np.prod([1 - (1 - sa) ** redundancy for sa in software_availability])
        system_result.append(system_unavail)
    return system_result

system_result_1 = calculate_system_result(service_avail_1)
system_result_2 = calculate_system_result(service_avail_2)
system_result_3 = calculate_system_result(service_avail_3)

# サービスの組み合わせを表示するためにラベルを作成
comb_labels = ['\n'.join([str(s) for s in comb]) for comb in all_combinations]
comb_labels_sparse = [comb_labels[i] if i % (2**(n-3)) == 0 else '' for i in range(len(comb_labels))]

# 散布図を作成
fig, ax = plt.subplots(figsize=(14, 8))

ax.scatter(comb_labels, system_result_1, color='blue', label='Service Availability 0.9')
ax.scatter(comb_labels, system_result_2, color='red', label='Service Availability 0.7(20%)&0.9(80%)')
ax.scatter(comb_labels, system_result_3, color='green', label='Service Availability 0.7(40%)&0.9(60%)')

ax.set_xlabel('Service Combinations', fontsize=14)
ax.set_ylabel('System Unavailability', fontsize=14)
ax.set_title(f'System Unavailability vs Service Combinations\n(n={n}, redundancy={redundancy}, h_add={h_add}, server_availability={server_avail})', fontsize=16)
ax.set_xticks(range(len(comb_labels_sparse)))
ax.set_xticklabels(comb_labels_sparse, rotation=90, fontsize=8)

# 可用性の分散を計算
variance_1 = np.var(system_result_1)
variance_2 = np.var(system_result_2)
variance_3 = np.var(system_result_3)

ax.text(0.02, 0.95, f'Variance 0.9: {variance_1:.2e}', transform=ax.transAxes, fontsize=12, verticalalignment='top', color='blue')
ax.text(0.02, 0.90, f'Variance 0.7(20%): {variance_2:.2e}', transform=ax.transAxes, fontsize=12, verticalalignment='top', color='red')
ax.text(0.02, 0.85, f'Variance 0.7(40%): {variance_3:.2e}', transform=ax.transAxes, fontsize=12, verticalalignment='top', color='green')
#ax.set_yscale('log')

ax.legend()
plt.tight_layout()
plt.show()
