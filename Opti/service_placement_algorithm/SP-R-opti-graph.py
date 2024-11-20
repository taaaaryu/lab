import time
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib
from itertools import combinations, chain, product
import random

# パラメータ
r_adds = [0.8, 1.0, 1.2]
num_service = [5,   10,20, 40, 60, 80, 100]
server_avail = 0.99
average = 10
max_redundancy = 4

# 各種関数
def calc_software_av_matrix(services_in_sw, service_avail, server_avail):
    services_array = np.array(services_in_sw, dtype=int)
    sw_avail_list = []
    count = 0
    for k in services_array:
        sw_avail = 1
        for i in range(k):
            sw_avail *= service_avail[count]
            count += 1
        sw_avail_list.append(sw_avail * server_avail)
    return sw_avail_list

def Greedy_Redundancy(sw_avail, sw_resource, H):
    num_sw = len(sw_avail)
    redundancy_list = [1] * num_sw
    sum_Resource = np.sum(sw_resource)
    sw_avail_list = sw_avail

    while sum_Resource <= H:
        sw_avail_sort, sw_resource, redundancy, sw_avail = zip(*sorted(zip(sw_avail_list, sw_resource, redundancy_list, sw_avail)))
        redundancy_list = list(redundancy)
        flag = 0
        for i in range(num_sw):
            if redundancy_list[i] >= max_redundancy:
                continue
            plus_resource = sw_resource[i]
            if (sum_Resource + plus_resource) <= H:
                redundancy_list[i] += 1
                sum_Resource += plus_resource
                sw_avail_list[i] = 1 - (1 - sw_avail[i]) ** int(redundancy_list[i])
                flag += 1
                break
        if flag == 0:
            break

    system_av = np.prod([1 - (1 - sa) ** int(r) for sa, r in zip(sw_avail, redundancy_list)])
    return redundancy_list, sum_Resource, system_av

# 結果の格納
results = {r_add: {"time_mean": [], "time_std": [], "unav_mean": [], "unav_std": []} for r_add in r_adds}

# 計算とデータ収集
for r_add in r_adds:
    for n in num_service:
        Resource = n * 2
        service_avail = [0.99] * n
        time_means = []
        unav_means = []

        for _ in range(average):
            start_time = time.time()
            services = list(range(1, n + 1))
            comb = [[services]]
            sw_resource = np.array([len(group) * (r_add ** (len(group) - 1)) for group in comb])
            sw_avail = calc_software_av_matrix([len(group) for group in comb], service_avail, server_avail)
            _, _, system_av = Greedy_Redundancy(sw_avail, sw_resource, Resource)
            end_time = time.time()
            time_means.append(end_time - start_time)
            unav_means.append(1 - system_av)

        results[r_add]["time_mean"].append(np.mean(time_means))
        results[r_add]["time_std"].append(np.std(time_means))
        results[r_add]["unav_mean"].append(np.mean(unav_means))
        results[r_add]["unav_std"].append(np.std(unav_means))

# データの整理
service_counts = np.array(num_service)
execution_time = {r_add: np.array(results[r_add]["time_mean"]) for r_add in r_adds}
execution_time_std = {r_add: np.array(results[r_add]["time_std"]) for r_add in r_adds}
unavailability = {r_add: np.array(results[r_add]["unav_mean"]) for r_add in r_adds}
unavailability_std = {r_add: np.array(results[r_add]["unav_std"]) for r_add in r_adds}

# グラフの作成
plt.figure(figsize=(14, 6))

# 実行時間のグラフ
plt.subplot(1, 2, 1)
for r_add in r_adds:
    plt.errorbar(service_counts, execution_time[r_add], yerr=execution_time_std[r_add], label=f"$\\mathrm{{r_{{add}}}}={r_add}$", capsize=5)
plt.xscale('linear')
plt.yscale('log')
plt.xlabel('サービス数', fontsize=14)
plt.ylabel('実行時間 (秒)', fontsize=14)
plt.legend()
plt.grid(True, which="both", linestyle='--', linewidth=0.5)

# 非可用性のグラフ
plt.subplot(1, 2, 2)
for r_add in r_adds:
    plt.errorbar(service_counts, unavailability[r_add], yerr=unavailability_std[r_add], label=f"$\\mathrm{{r_{{add}}}}={r_add}$", capsize=5)
plt.xscale('linear')
plt.yscale('log')
plt.xlabel('サービス数', fontsize=14)
plt.ylabel('非可用性', fontsize=14)
plt.legend()
plt.grid(True, which="both", linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig("提案手法_実行時間_非可用性_エラーバー付き.png")
plt.show()