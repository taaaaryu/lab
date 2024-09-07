import time
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations, chain, product
import random
# パラメータ
Resource = [20]  # サーバリソース
r_adds= [1]  # サービス数が1増えるごとに使うサーバ台数の増加


# 定数
n = 10  # サービス数
softwares = [i for i in range(1, n+1)]
services = [i for i in range(1, n + 1)]
service_avail = [0.99]*n
#service_avail = [0.9, 0.99, 0.99, 0.99, 0.99, 0.9, 0.99, 0.99, 0.99, 0.99]
server_avail = 0.99
NUM_START = [50]
NUM_NEXT = [5]
GENERATION = 10
average = 1

max_redundancy = 5

# ソフトウェアの可用性を計算する関数
def calc_software_av(services_group, service_avail,services):
    indices = [services.index(s) for s in services_group]
    result = 1.0
    for i in indices:
        result *= service_avail[i]
    return result

def calc_software_av_matrix(services_in_sw, service_avail, server_avail):
    services_array = np.array(services_in_sw, dtype=int)
    sw_avail_list = []
    count = 0
    for k in services_array:
        sw_avail=1
        for i in range(k):
            sw_avail *= service_avail[count]
            count += 1
        sw_avail_list.append(sw_avail*server_avail)
    return sw_avail_list

def generate_service_combinations(services, num_software):
    all_combinations = []
    n = len(services)
    for indices in combinations(range(n - 1), num_software - 1):
        split_indices = list(chain([-1], indices, [n - 1]))
        combination = [services[split_indices[i] + 1: split_indices[i + 1] + 1] for i in range(len(split_indices) - 1)]
        all_combinations.append(combination)
    return all_combinations

def generate_redundancy_combinations(num_software):
    red_comb = []
    for i in range(1,num_software+1):
        red_comb.append(product(range(1, max_redundancy), repeat=i))
    return red_comb

def greedy_search(matrix, software_count, service_avail, server_avail, r_add, H):
    best_RUEs = [-np.inf]*num_next
    best_matrices = [None]*num_next
    best_counts = [0]*num_next

    list = []
    best_matrix = matrix.copy()
    best_RUE = calc_RUE(matrix, software_count, service_avail, server_avail, r_add, H)

    for k in range(GENERATION):
        RUE_list = [best_RUE]
        matrix = best_matrix.copy()
        one_list = []
        col = 0
        for i in range(len(matrix[0])):
            if matrix[col][i] == 0:
                one_list.append(i)
                col += 1

        mini_RUE_list = [0]
        matrix_list = [[0]]
        for j in range(len(one_list)):
            a = one_list[j]
            one = matrix.copy()
            one[j][a - 1] = 0
            one[j][a] = 1

            two = matrix.copy()
            two[j][a - 1] = 1
            two[j][a] = 0

            one_new_RUE = calc_RUE(one, software_count, service_avail, server_avail, r_add, H)
            mini_RUE_list.append(one_new_RUE)
            matrix_list.append(one)
            two_new_RUE = calc_RUE(two, software_count, service_avail, server_avail, r_add, H)
            mini_RUE_list.append(two_new_RUE)
            matrix_list.append(two)

        new_RUE = max(mini_RUE_list)
        idx = mini_RUE_list.index(new_RUE)
        new_matrix = matrix_list[idx]
        RUE_list.append(new_RUE)

        one_list.append(len(matrix[0]))
        one_list.insert(0, 0)

        if software_count <= n-1:
            new_sw_p_matrix = divide_sw(matrix, one_list)
            new_RUE_p = calc_RUE(new_sw_p_matrix, len(new_sw_p_matrix), service_avail, server_avail, r_add, H)
            RUE_list.append(new_RUE_p)
        else:
            new_RUE_p = 0

        if software_count >= 2:
            new_sw_n_matrix = integrate_sw(matrix, one_list)
            new_RUE_n = calc_RUE(new_sw_n_matrix, len(new_sw_n_matrix), service_avail, server_avail, r_add, H)
            RUE_list.append(new_RUE_n)
        else:
            new_RUE_n = 0

        max_RUE = max(RUE_list)
            
        if max_RUE > best_RUE:
            if max_RUE == new_RUE:
                best_RUE = new_RUE
                best_matrix = new_matrix
            elif max_RUE == new_RUE_p:
                best_RUE = max_RUE
                best_matrix = new_sw_p_matrix
                software_count += 1
            elif max_RUE == new_RUE_n:
                best_RUE = max_RUE
                best_matrix = new_sw_n_matrix
                software_count -= 1
        else:
            best_RUE = max_RUE
        list.append(best_RUE)

                
        if best_RUE > best_RUEs[0]:
            # 新しいRUEが現在のトップよりも高い場合、他のRUEを下にシフトして、新しいRUEを最上位に挿入する
            for i in range(num_next - 1, 0, -1):
                best_RUEs[i] = best_RUEs[i - 1]
                best_matrices[i] = best_matrices[i - 1]
                best_counts[i] = best_counts[i - 1]

            best_RUEs[0] = best_RUE
            best_matrices[0] = best_matrix
            best_counts[0] = software_count
        else:
            for i in range(1, num_next):
                if best_RUE > best_RUEs[i]:
                    # 指定した位置に新しいRUEを挿入し、他のRUEを下にシフトする
                    for j in range(num_next - 1, i, -1):
                        best_RUEs[j] = best_RUEs[j - 1]
                        best_matrices[j] = best_matrices[j - 1]
                        best_counts[j] = best_counts[j - 1]

                    best_RUEs[i] = best_RUE
                    best_matrices[i] = best_matrix
                    best_counts[i] = software_count
                    break

    return best_matrices, best_counts, best_RUEs, list

def calc_RUE(matrix, software_count, service_avail, server_avail, r_add, H):
    avg_efficiency = []
    initial_redundancy = np.ones(software_count)
    sum_matrix = np.sum(matrix, axis=1)
    software_availability = calc_software_av_matrix(sum_matrix, service_avail, server_avail)
    sw_list = np.array(software_availability)
    system_avail = np.prod(sw_list)
    matrix_resource = r_add * (sum_matrix - 1) + 1
    total_servers = np.dot(initial_redundancy, matrix_resource)  # dot product

    for i in range(software_count):
        initial_redundancy[i] += 1
        total_servers_red = np.dot(initial_redundancy, matrix_resource)
        total_servers_mask = total_servers_red <= H
        system_avail_red = np.prod([1 - (1 - sa) ** int(r) for sa, r in zip(software_availability, initial_redundancy)])
        redundancy_cost_efficiency = np.where(total_servers_mask, (system_avail_red - system_avail) / (total_servers_red - total_servers), 0)
        avg_efficiency.append(redundancy_cost_efficiency) # 1 corresponds to redundancy=2
        initial_redundancy[i] = 1
    return np.mean(avg_efficiency)

def multi_start_greedy(r_add, service_avail, server_avail, H, num_service, num_starts):
    best_global_matrices = [None] * num_next
    best_global_RUEs = [-np.inf] * num_next
    best_global_counts = [0] * num_next
    RUE_list = []
    x_gene = np.arange(1, GENERATION + 1)
    service = np.arange(1, num_service)
    software_count_float = np.random.normal(num_service / 2, 2, num_starts)
    software_counts = np.clip(software_count_float.astype(int), 1, n)

    for software_count in software_counts:
        matrix = make_matrix(service, software_count)
        best_matrices, best_counts, best_RUEs, RUE_each_list = greedy_search(matrix, software_count, service_avail, server_avail, r_add, H)

        RUE_list.append(RUE_each_list)

        for i in range(num_next):
            if best_RUEs[i] > best_global_RUEs[i]:
                if all(not np.array_equal(best_matrices[i], bm) for bm in best_global_matrices):
                    best_global_matrices[i] = best_matrices[i]
                    best_global_counts[i] = best_counts[i]
                    best_global_RUEs[i] = best_RUEs[i]

    return best_global_matrices, best_global_counts, best_global_RUEs

def make_matrix(service, software_count):
    matrix = np.zeros((software_count, len(service) + 1), dtype=int)
    service_list = service.tolist()
    a = random.sample(service_list, software_count - 1)
    a.append(len(service) + 1)
    a.sort()
    idx = 0
    for i in range(software_count):
        for k in range(idx, a[i]):
            matrix[i][k] = 1
            idx += 1
    return matrix

def divide_sw(matrix, one_list):
    flag = 0
    cp_list = one_list.copy()
    while flag == 0:
        idx = random.randint(0, len(cp_list) - 2)
        start = cp_list[idx]
        end = cp_list[idx + 1]
        if end - start > 1:
            a = random.randint(start + 1, end - 1)
            div_matrix = np.insert(matrix, idx + 1, 0, axis=0)
            for i in range(a, cp_list[idx + 1]):
                div_matrix[idx][i] = 0
                div_matrix[idx + 1][i] = 1
            flag = 1
        else:
            continue
    return div_matrix

def integrate_sw(matrix, one_list):
    cp_list = one_list.copy()
    idx = random.randint(1, len(cp_list) - 2)
    start = cp_list[idx - 1]
    end = cp_list[idx + 1]
    for i in range(start, end):
        matrix[idx - 1][i] = 1
    new_matrix = np.delete(matrix, idx, 0)
    return new_matrix

def find_ones(matrix):
    # numpy配列に変換
    arr = np.array(matrix)
    
    # 1の位置を探す
    rows, cols = np.nonzero(arr)
    
    # 行ごとに1のある列インデックスをまとめる
    positions = [[col + 1 for col in cols[rows == row]] for row in np.unique(rows)]
    
    return positions


for num_starts in NUM_START:
    for num_next in NUM_NEXT:
        unav_list = []
        time_list = []
        for r_add in r_adds:
            for H in Resource:
                alloc = H*0.9  #サーバリソースの下限
                time_mean = []
                unav_mean = []

                for i in range(average):
                    start = time.time()
                        #fig, ax = plt.subplots(figsize=(12, 8))
                    best_matrix, best_software_count, best_RUE = multi_start_greedy(r_add, service_avail, server_avail, H, len(services),num_starts)

                    min_unav = []
                    
                    best_combinations = []
                    all_redundancies = generate_redundancy_combinations(n)
                    

                    for p in best_matrix:
                        if p is not None:
                            best_combinations.append(find_ones(p))

                    placement_result = []
                    p_results = []

                    for comb in best_combinations:

                        max_system_avail = -1
                        best_redundancy = None
                        # software_availability の計算をループ外に移動
                        software_availability = [calc_software_av(group, service_avail, services)*server_avail for group in comb]
                        sw_resource = np.array([r_add * (len(group) - 1) + 1 for group in comb])

                        for redundancy in all_redundancies[len(comb)-1]:
                            red_array = np.array(redundancy)
                            sw_red_resource = sw_resource * red_array
                            total_servers = np.sum(sw_red_resource)
                            if total_servers <= H:
                                if alloc <= total_servers:
                                    # 最適化されたsystem_avail計算
                                    system_avail = np.prod([1 - (1 - sa) ** int(r) for sa, r in zip(software_availability, redundancy)])
                                    if system_avail > max_system_avail:
                                        max_system_avail = system_avail
                                        best_redundancy = redundancy

                        if best_redundancy:
                            p_results.append((comb, best_redundancy, max_system_avail))
                            max_avails = [max_avail for _, _, max_avail in p_results]
                            print(p_results)
                            placement_result.append(max(max_avails))
                    end = time.time()
                    
                    time_diff = end - start
                
                    time_mean.append(time_diff)
                    unav_mean.append(1-max(placement_result))

                time_list.append(sum(time_mean)/len(time_mean))
                unav_list.append(np.sum(unav_mean)/len(unav_mean))
                
        print(f"{num_starts}-{num_next}-result")
        print("time")
        for i in range(len(r_adds)*len(Resource)):
            print(time_list[i])
        print("unav")
        for i in range(len(r_adds)*len(Resource)):
            print(unav_list[i])


