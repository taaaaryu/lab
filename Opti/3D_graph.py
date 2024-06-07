import time
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations, chain
from mpl_toolkits.mplot3d import Axes3D

# 定数
n = 10  # サービス数
services = [i for i in range(1, n + 1)]
service_avail = [0.9]*n
server_avail = 0.9
h_add = 0.5  # サービス数が1増えるごとに使うサーバ台数の増加
H = 20  # 全体のリソース
redundancy = 5  # 最大の冗長化数

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
num_comb = len(all_combinations)

# 冗長度合いを計算し、システム非可用性を記録
system_result = []

for r in range(1, redundancy + 1):
    one_result = []
    for comb in all_combinations:
        software_availability = [calc_software_av(group) * server_avail for group in comb]
        system_unavail = 1 - np.prod([1 - (1 - sa) ** r for sa in software_availability])
        one_result.append(system_unavail)
    system_result.append(one_result)

# サービスの組み合わせを表示するためにラベルを作成
comb_labels = ['\n'.join([str(s) for s in comb]) for comb in all_combinations]
comb_labels_sparse = [comb_labels[i] if i % (2**(n-3)-1) == 0 else '' for i in range(len(comb_labels))]

# x軸を冗長化度、y軸をサービスの組み合わせ、z軸をシステムの非可用性としてプロット
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

X = np.arange(1, num_comb + 1)
Y = np.arange(1, redundancy + 1)
X, Y = np.meshgrid(X, Y)
Z = np.array(system_result)

ax.plot_surface(Y, X, Z, cmap='viridis')

# Y軸の目盛りをサービスの組み合わせに変更
ax.set_xticks(np.arange(1, redundancy + 1))
ax.set_yticks(np.arange(1, num_comb + 1))
ax.set_yticklabels(comb_labels_sparse, fontsize=8, rotation=0, va='center', ha='right', rotation_mode='anchor')

# 各冗長化数におけるすべてのサービス組み合わせのシステム可用性を黒線で追加
for y in range(1, redundancy + 1):
    ax.plot(np.full(num_comb, y), np.arange(1, num_comb + 1), Z[y-1, :], color='k',  linestyle='-', linewidth=1, zorder=10)

# z軸の表示範囲を設定
ax.set_zlim(1e-5, 9e-1)

ax.set_xlabel('Redundancy Level')
ax.set_ylabel('Service Combinations')
ax.set_zlabel('System Unavailability (log scale)')
ax.set_zscale('log')
ax.set_title(f'num of service={n}, server_availability = {server_avail}, h_add = {h_add}, resource = 20')

ax.view_init(elev=20, azim=30)  # elev: 上下の角度, azim: 水平の角度
plt.show()
