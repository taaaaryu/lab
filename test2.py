import time
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations, chain, product
from matplotlib.colors import to_rgba

# 定数
n = 10  # サービス数
num_software = 5
services = [i for i in range(1, n + 1)]
service_avail = [0.99]*n
#service_avail = [0.9, 0.99, 0.99, 0.99, 0.99, 0.9, 0.99, 0.99, 0.99, 0.99]
server_avail = 0.99
H = 20  # サーバの台数

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



for h_add in h_add_values:
    all_combinations = generate_service_combinations(services, num_software)
    all_redundancies = generate_redundancy_combinations(num_software, H, h_add)
    print(f"comb={len(all_combinations)}, red={len(all_redundancies)}")

