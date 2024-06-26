import time
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations, chain, product

# 定数
n = 10  # サービス数
num_software = 3
services = [i for i in range(1, n + 1)]
service_avail = [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
server_avail = 0.99
H = 20  # サーバの台数
h_add = 1  # サービス数が1増えるごとに使うサーバ台数の増加

start_time = time.time()  # 処理開始時間

# ソフトウェアの可用性を計算する関数
def calc_software_av(services_group):
    indices = [services.index(s) for s in services_group]
    return np.prod([service_avail[i] for i in indices])

# サービスの組み合わせを生成する関数
def generate_service_combinations(services):
    all_combinations = []
    n = len(services)
    for indices in combinations(range(n - 1), num_software - 1):
        split_indices = list(chain([-1], indices, [n - 1]))
        combination = [services[split_indices[i] + 1: split_indices[i + 1] + 1] for i in range(len(split_indices) - 1)]
        all_combinations.append(combination)
    return all_combinations

# サービスのすべての組み合わせを生成
all_combinations = generate_service_combinations(services)

# 冗長化の組み合わせを生成する関数
def generate_redundancy_combinations(num_software, max_servers, h_add):
    all_redundancies = []
    for redundancy in product(range(1, max_servers // h_add + 1), repeat=num_software):
        if sum(redundancy) * h_add <= max_servers:
            all_redundancies.append(redundancy)
    return all_redundancies

# 冗長化のすべての組み合わせを生成
all_redundancies = generate_redundancy_combinations(num_software, H, h_add)

# 冗長化の組み合わせごとにシステム可用性を計算し、最も高いシステム可用性を探索
results = []

for redundancy in all_redundancies:
    max_system_avail = -1
    best_combination = None
    for comb in all_combinations:
        total_servers = sum(redundancy[i] * len(comb[i]) for i in range(len(comb)))
        if total_servers <= H:
            software_availability = [calc_software_av(group) * server_avail for group in comb]
            system_avail = np.prod([1 - (1 - sa) ** int(r) for sa, r in zip(software_availability, redundancy)])

            if system_avail > max_system_avail:
                max_system_avail = system_avail
                best_combination = comb
    if best_combination:
        results.append((redundancy, best_combination, max_system_avail))

# プロットに必要なデータを確認
for redundancy, comb, max_avail in results:
    print(f"Redundancy: {redundancy}, Best Combination: {comb}, Max System Availability: {max_avail}")


max_avails = [max_avail for _, _, max_avail in results]
system_av_forCDF = sorted(max_avails)
N = len(system_av_forCDF)
sy = [i/N for i in range(N)]

# プロットを作成
fig, ax = plt.subplots(figsize=(12, 8))

#combinations_labels = ['\n'.join(map(str, comb)) for comb, _ in comb_max_system_unavail]
#max_avails = [max_avail for _, max_avail in comb_max_system_unavail]

ax.plot(system_av_forCDF,sy, color='b', alpha=0.7)

# ラベルを追加
ax.set_xlabel('System Availability')
ax.set_ylabel('CDF')
#ax.set_title('System Availability for Different Service Combinations')

ax.set_xscale('log')


plt.xticks(rotation=45, ha='right')
plt.tight_layout()


plt.show()

