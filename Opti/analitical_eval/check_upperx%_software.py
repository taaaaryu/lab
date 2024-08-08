import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import numpy as np
from itertools import combinations, chain, product
from matplotlib.colors import to_rgba
import statistics

# パラメータ
n = 10  # サービス数
H = n*2  # サーバリソース
h_add = 1.0  # サービス数が1増えるごとに使うサーバ台数の増加

# 定数

softwares = [i for i in range(1, n+1)]
services = [i for i in range(1, n + 1)]
service_avail = [0.99]*n
#service_avail = [0.99,0.99,0.99,0.9,0.99,0.99,0.9,0.99,0.99,0.99]
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

def calc_sw_resource(comb):
    sw = [(h_add * (len(i) - 1)) + 1 for i in comb]
    return sw

software_availabilities = []
combination_lengths = []
results = []


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
            results.append((redundancy, best_combination, max_system_avail))
            combination_lengths.append((max_system_avail, len(best_combination)))
        progress_tqdm.update(1)

progress_tqdm.close()

# 上位x%のソフトウェア数を計算
top_x_count = int(top_x_percent*len(combination_lengths))

# 上位x%のシステム可用性に対応するbest_combinationの長さの分布
results.sort(key=lambda x: x[2], reverse=True)
combination_lengths.sort(key=lambda x: x[0], reverse=True)
std_sw_resource = []

#print(results[:20])
comb = [comb for red, comb, avail in results[:top_x_count]]
each_sw_resource = [calc_sw_resource(k) for k in comb]
for j in each_sw_resource:
    if len(j)==1:
        continue
    std_sw_resource.append(statistics.stdev(j))
print(std_sw_resource)
top_combination_lengths = [length for avail, length in combination_lengths[:top_x_count]]
#print(top_combination_lengths)

count_software = [top_combination_lengths.count(i) for i in range(1,num_software+1)]

# 標準偏差のCDFを計算し、プロット
std_sw_resource.sort()
cdf = np.arange(len(std_sw_resource)) / float(len(std_sw_resource) - 1)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(std_sw_resource, cdf, marker='o', linestyle='-', color='b')
ax.set_xlabel('Standard Deviation of SW Resource')
ax.set_ylabel('CDF')
ax.set_title('CDF of Standard Deviation of SW Resource')
plt.grid(True)
plt.show()

# ソフトウェア数の分布をプロット
fig, ax = plt.subplots(figsize=(12, 8))
x = [i for i in range(1,num_software+1)]

ax.bar(x, count_software, edgecolor='black')
ax.set_xlabel('Number of Software')
ax.set_ylabel('Counts')
ax.set_title(f'Top {top_x_percent*100}% Availability')
ax.set_xticks(x)

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
