import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import product
import time

def calc_software_av(services_in_sw, service_avail, server_avail):
    non_sero_idx = np.nonzero(services_in_sw)
    sw_avail = 1
    non_sero_idx = [item.tolist() for item in non_sero_idx]
    for k in non_sero_idx[0]:
        sw_avail *= service_avail[k] * services_in_sw[k]
    return sw_avail * server_avail

def make_matrix(service, software_count):
    matrix = np.zeros((software_count, len(service) + 1), dtype=int)
    a = random.sample(service, software_count - 1)
    a.append(len(service) + 1)
    a.sort()
    idx = 0
    for i in range(software_count):
        for k in range(idx, a[i]):
            matrix[i][k] = 1
            idx += 1
    return matrix

def calc_RUE(matrix, software_count, service_avail, server_avail, r_add, H):
    initial_redundancy = [1] * software_count
    redundancy_cost_efficiency = []
    software_availability = [calc_software_av(matrix[j], service_avail, server_avail) for j in range(software_count)]
    system_avail = np.prod([1 - (1 - sa) ** int(r) for sa, r in zip(software_availability, initial_redundancy)])
    total_servers = sum(initial_redundancy[j] * ((r_add * (np.sum(matrix[j]) - 1)) + 1) for j in range(software_count))

    for i in range(software_count):
        initial_redundancy[i] = 2
        total_servers_red = sum(initial_redundancy[j] * ((r_add * (sum(matrix[j]) - 1)) + 1) for j in range(software_count))

        if total_servers_red <= H:
            system_avail_red = np.prod([1 - (1 - sa) ** int(r) for sa, r in zip(software_availability, initial_redundancy)])
            redundancy_cost_efficiency.append((system_avail_red - system_avail) / (total_servers_red - total_servers))
        else:
            redundancy_cost_efficiency.append(0)

        initial_redundancy[i] = 1

    avg_efficiency = np.mean(redundancy_cost_efficiency)
    return avg_efficiency

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

# Function to generate redundancy combinations
def generate_redundancy_combinations(num_software, max_servers, r_add):
    all_redundancies = [redundancy for redundancy in product(range(1, max_redundancy), repeat=num_software)]
    return all_redundancies

def search_best_redundancy(comb, all_redundancy): #all_combinationsは
    r_unav = []
    best_reds = []
    best_comb = []
    max_system_avail = -1
    best_redundancy = None
    sw = len(comb)
    for redundancy in all_redundancy:
        total_servers = sum(redundancy[i] * ((r_add*(sum(comb[i])-1))+1) for i in range(sw))
        if total_servers <= H:
            if alloc*H <= total_servers:
                software_availability = [calc_software_av(group, service_avail, server_avail) for group in comb]
                system_avail = np.prod([1 - (1 - sa) ** int(r) for sa, r in zip(software_availability, redundancy)])
                if system_avail > max_system_avail:
                    max_system_avail = system_avail
                    best_redundancy = redundancy
    if best_redundancy:
        r_unav.append(1 - max_system_avail)
        best_reds.append(best_redundancy)
        best_comb.append(comb)

    return r_unav, best_reds, best_comb


def greedy_search(matrix, software_count, service_avail, server_avail, r_add, H, service):
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

        mini_RUE_list = []
        matrix_list = []
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
        idx = mini_RUE_list.index(best_RUE)
        new_matrix = matrix_list[idx]
        RUE_list.append(new_RUE)

        one_list.append(len(matrix[0]))
        one_list.insert(0, 0)

        if software_count <= 9:
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

def multi_start_greedy(r_add, service_avail, server_avail, H, num_service, num_starts):
    best_global_matrices = [None]*num_next
    best_global_RUEs = [-np.inf]*num_next
    best_global_counts = [0]*num_next
    RUE_list = []
    x_gene = [i for i in range(1, GENERATION + 1)]
    service = [i for i in range(1, num_service)]
    software_count_float = np.random.normal(num_service/2, 2, num_starts)
    software_counts = [int(round(software_count_float[n], 0)) for n in range(len(software_count_float))] 

    for k in range(num_starts):
        software_count = software_counts[k]
        if software_count < 1 or 10 < software_count:
            if software_count < 1:
                software_count = 1
            else:
                software_count = 10

        matrix = np.zeros((software_count, num_service), dtype=int)
        if software_count == 1:
            matrix = np.ones((software_count, num_service), dtype=int)
            continue
        else:
            matrix = make_matrix(service, software_count)

        best_matrices, best_counts, best_RUEs, RUE_each_list = greedy_search(matrix, software_count, service_avail, server_avail, r_add, H, service)

        RUE_list.append(RUE_each_list)

        for i in range(num_next):
            if best_RUEs[i] > best_global_RUEs[i]:
                if all(np.array_equal(best_matrices[i], bm) == False for bm in best_global_matrices):
                    best_global_matrices[i] = best_matrices[i]
                    best_global_counts[i] = best_counts[i]
                    best_global_RUEs[i] = best_RUEs[i]
        
        plt.plot(x_gene, RUE_each_list)

    return best_global_matrices, best_global_counts, best_global_RUEs


# 使用例
r_adds = [0.5]  # 例としてr_add値
num_service = 10 #サービス数
service_avail = [0.99]*num_service  # サービス可用性の例
server_avail = 0.99  # サーバー可用性の例
Resources = [15,20,25]  # 最大サーバー制約の例
max_redundancy = 5 #1つのSWの冗長化度合い上限
num_starts = 30
num_next = 10 #何個を冗長化するか
alloc = 0.95

GENERATION = 10

for H in Resources:
    for r_add in r_adds:
        start = time.time()
        fig, ax = plt.subplots(figsize=(12, 8))
        best_matrix, best_software_count, best_RUE = multi_start_greedy(r_add, service_avail, server_avail, H, num_service,num_starts)

        plt.xlabel("Generation")
        plt.ylabel("RCE")
        plt.title(f"r_add = {r_add}, Resource = {H}")
        
        #plt.show()
        #plt.savefig(f"RCE-Greedy_{r_add}-{H}.png", bbox_inches='tight', pad_inches=0)
        print(f"r_add = {r_add}, Resource = {H}")
        
        result_unav = []
        result_red = []
        result_comb = []
        
        """for i in range(num_next):
            print(f"Best Matrix:\n{best_matrix[i]}")
            print(f"Best Software Count: {best_software_count[i]}")
            print(f"Best RCE: {best_RUE[i]}")"""
        
        for p in range(num_next):
            if best_matrix[p] is not None:
                sw_redundancies = generate_redundancy_combinations(best_software_count[p], H, r_add)
                unav, red, comb = search_best_redundancy(best_matrix[p], sw_redundancies)
                result_unav.append(unav)
                result_comb.append(comb)
                result_red.append(red)
        min_unav = min(result_unav)
        opti_idx = result_unav.index(min_unav)
        opti_comb = result_comb[opti_idx]
        opti_red = result_red[opti_idx]
        end = time.time()
        
        time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する
        print(f"time = {time_diff}")  # 処理にかかった時間データを使用
        
        print(f"minimum Uavailability = {min_unav}")
        print(opti_comb,opti_red)
        
            
        
            
