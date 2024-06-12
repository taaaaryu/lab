import time
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations, chain
from mpl_toolkits.mplot3d import Axes3D

# 定数
n = 10  # サービス数
services = [i for i in range(1, n + 1)]
service_avail = [0.99] * n
server_avail = 0.99
h_add = 1  # サービス数が1増えるごとに使うサーバ台数の増加
server_resources = [25, 26, 27, 28, 29, 30]  # 複数のサーバリソースを設定

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

# システム非可用性を格納するリスト
system_unavailabilities = []

max_length = 0  # 最大の長さを追跡する変数

for H in server_resources:
    temp_unavailabilities = []
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
            temp_unavailabilities.append(1 - system_avail)

            # システム可用性が最も上昇するように1つのソフトウェアを冗長化
            improvements = []
            for i in range(len(comb)):
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

    max_length = max(max_length, len(temp_unavailabilities))  # 最大の長さを更新
    system_unavailabilities.append(temp_unavailabilities)

# 最大の長さに揃えるためにパディング
for unavailabilities in system_unavailabilities:
    while len(unavailabilities) < max_length:
        unavailabilities.append(np.nan)

# 面グラフを作成
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# X軸はサービスの組み合わせ、Y軸はサーバリソース、Z軸はシステム非可用性
X = np.arange(1, max_length + 1)
Y = np.array(server_resources)
X, Y = np.meshgrid(X, Y)
Z = np.array(system_unavailabilities)

ax.plot_surface(X, Y, Z, cmap='viridis')

# 各リソースにおけるシステム非可用性を黒線で追加
for i in range(len(server_resources)):
    ax.plot(X[i], [server_resources[i]] * max_length, Z[i], color='k', linewidth=1)

# 軸の設定
ax.set_xlabel('Service Combinations')
ax.set_ylabel('Server Resources (H)')
ax.set_zlabel('System Unavailability')
ax.set_zscale('log')
ax.set_title('System Unavailability for Different Configurations')

# グラフの視点を調整
ax.view_init(elev=30, azim=60)

# カラーバーを追加
mappable = ax.plot_surface(X, Y, Z, cmap='viridis')
fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)

# 注釈を追加
for i in range(len(server_resources)):
    max_unavail = np.nanmax(Z[i])
    max_idx = np.nanargmax(Z[i])
    ax.text(X[i, max_idx], Y[i, max_idx], Z[i, max_idx], f'{max_unavail:.2e}', color='red')

end_time = time.time()  # 処理終了時間
elapsed_time = end_time - start_time  # 経過時間
print(f"Elapsed time: {elapsed_time:.2f} seconds")

plt.show()
