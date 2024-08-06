import time
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations, chain
from matplotlib.colors import to_rgba

# 定数
n = 15  # サービス数
services = [i for i in range(1, n + 1)]
service_avail = [0.7, 0.8 ,0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
server_avail = 0.99
server_resource = [i for i in range(1, 31)]  # サーバの台数
h_add = 1.0  # サービス数が1増えるごとに使うサーバ台数の増加

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

# 冗長度合いを計算し、システム可用性が最大となる条件を探す
max_system_avail = -1
best_combination = None
best_redundancy = None
best_num_r = None

h_result = []
comb_result = []
redundancy_result = []
num_r_result = []

for H in server_resource:
    for comb in all_combinations:
        redundancy = [1] * len(comb)  # 冗長化してないときのソフトウェアごとに使用するサーバリソースを計算
        num_r = [1] * len(comb)
        for i in range(len(comb)):
            redundancy[i] += (len(comb[i]) - 1) * h_add
        total_servers = sum(redundancy)
        base_redundancy = redundancy.copy()

        while total_servers <= H:
            software_availability = [calc_software_av(group) * server_avail for group in comb]
            system_avail = np.prod([1 - (1 - sa) ** int(r) for sa, r in zip(software_availability, redundancy)])

            if system_avail > max_system_avail:
                max_system_avail = system_avail
                best_combination = comb
                best_redundancy = redundancy.copy()
                best_num_r = num_r.copy()

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
                new_system_avail = np.prod([1 - (1 - sa) ** int(r) for sa, r in zip(new_software_availability, new_redundancy)])
                improvement = new_system_avail - system_avail
                improvements.append((improvement, i, new_redundancy, new_num_r))

            if not improvements:
                break

            # 最大の改善をもたらす冗長化を適用
            best_improvement = max(improvements)
            redundancy = best_improvement[2]
            num_r = best_improvement[3]
            total_servers = sum(redundancy)

    h_result.append(1 - max_system_avail)
    comb_result.append(best_combination)
    redundancy_result.append(best_redundancy)  # redundancy = 各ソフトウェアが消費するリソース
    num_r_result.append(best_num_r)  # num_r = 各ソフトウェアの冗長度

for k in range(len(h_result))[::-1]:
    if h_result[k] == 2:
        del h_result[k]
        del server_resource[k]
        del comb_result[k]
        del redundancy_result[k]
        del num_r_result[k]

# 各サービスがどのソフトウェアに内包されているかをマッピング
service_to_software = {}
if best_combination:
    for idx, software in enumerate(best_combination):
        for service in software:
            service_to_software[service] = f'Software {idx + 1}'

# プロットを作成
fig, ax = plt.subplots(figsize=(14, 7))
ax2 = ax.twinx()

xl = 'Server Resource'
yl = 'System Unavailability'

ax.plot(server_resource, [unavail for unavail in h_result], linestyle="solid", color='r', marker='o', label="System Unavailability")

# ソフトウェアの冗長度合いを積み上げ棒グラフとして描画
colors = plt.cm.tab20.colors
max_redundancy = max(max(row) for row in num_r_result)
for i in range(len(server_resource)):
    bottom = 0
    for k in range(len(comb_result[i])):
        redundancy_level = num_r_result[i][k]
        opacity = 0.2 + 0.8 * (redundancy_level / max_redundancy)
        color_with_opacity = to_rgba(colors[k % len(colors)], opacity)
        bar = ax2.bar(server_resource[i], redundancy_result[i][k], bottom=bottom, color=color_with_opacity, label=f'Software {k+1}' if i == 0 else "")
        text_x = server_resource[i]
        text_y = bottom + redundancy_result[i][k]
        ax2.text(text_x, text_y, f'{num_r_result[i][k]}', ha='center', va='bottom', color='black', fontsize=8)
        ax2.text(text_x, text_y, f'\n{comb_result[i][k]}', ha='center', va='top', color='black', fontsize=6)
        bottom += redundancy_result[i][k]

ax2.set_ylabel('Software Resource')

# ラベルを追加
ax.set_xlabel(xl)
ax.set_ylabel(yl)

ax.set_zorder(ax2.get_zorder() + 1)
ax.set_frame_on(False)

ax.set_yscale('log')
plt.title(f'System Unavailability - Server Resource (number of service = {len(services)}, h_add = {h_add})')

plt.grid(True)

end_time = time.time()  # 処理終了時間
elapsed_time = end_time - start_time  # 経過時間

print("Elapsed time: {:.2f} seconds".format(elapsed_time))

plt.show()
