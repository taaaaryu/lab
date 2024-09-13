import time
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations, chain, product
# パラメータ
Resource = [20]  # サーバリソース
h_adds= [1]  # サービス数が1増えるごとに使うサーバ台数の増加


# 定数
n = 5  # サービス数
softwares = [i for i in range(1, n+1)]
services = [i for i in range(1, n + 1)]
service_avail = [0.99]*n
#service_avail = [0.9, 0.99, 0.99, 0.99, 0.99, 0.9, 0.99, 0.99, 0.99, 0.99]
server_avail = 0.99

max_redundancy = 5

# ソフトウェアの可用性を計算する関数
def calc_software_av(services_group, service_avail,services):
    indices = [services.index(s) for s in services_group]
    result = 1.0
    for i in indices:
        result *= service_avail[i]
    return result


def generate_service_combinations(services, num_software):
    all_combinations = []
    n = len(services)
    for indices in combinations(range(n - 1), num_software - 1):
        split_indices = list(chain([-1], indices, [n - 1]))
        combination = [services[split_indices[i] + 1: split_indices[i + 1] + 1] for i in range(len(split_indices) - 1)]
        all_combinations.append(combination)
    return all_combinations

# 冗長化の組み合わせを生成する関数

def generate_redundancy_combinations(num_software):
    return product(range(1, max_redundancy), repeat=num_software)

# プロットを作成
fig, ax = plt.subplots(2,1,figsize=(12, 8))

software_result = []
unav_list = []
time_list = []

for r_add in h_adds:
    for H in Resource:
        alloc = H*0.95 #サーバリソースの下限
        start = time.time()
        placement_result = []
        
        for num_software in softwares:
            all_combinations = generate_service_combinations(services, num_software)

            # サービス実装形態によるCDFの計算
            p_results = []
            for comb in all_combinations:
                num_software = len(comb)

                max_system_avail = -1
                best_redundancy = None

                # software_availability の計算をループ外に移動
                software_availability = [calc_software_av(group, service_avail, services)*server_avail for group in comb]
                sw_resource = np.array([r_add * (len(group) - 1) + 1 for group in comb])
                
                
                #ここから冗長化アルゴリズムをかく
                start_red = [1]*num_software
                






        end = time.time()
        
        time_diff = end - start
        print(f"time = {time_diff}")
        
        time_list.append(time_diff)
        unav_list.append(1-max(placement_result))
for i in range(len(h_adds)*len(Resource)):
    print(time_list[i])
for i in range(len(h_adds)*len(Resource)):
    print(unav_list[i])


