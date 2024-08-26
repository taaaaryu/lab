import numpy as np
import random
import matplotlib.pyplot as plt

def calc_software_av(services_in_sw, service_avail,server_avail):
    # 各SWの可用性を計算
    non_sero_idx = np.nonzero(services_in_sw)
    sw_avail=1
    non_sero_idx= [item.tolist() for item in non_sero_idx]

    for k in non_sero_idx[0]:
        sw_avail *= service_avail[k]*services_in_sw[k]
    return sw_avail*server_avail

def make_matrix(service,software_count):
    matrix = np.zeros((software_count, len(service)), dtype=int)
    a = random.sample(service,software_count-1)
    a.append(len(service))
    a.sort()
    idx = 0
    for i in range(software_count):
        for k in range(idx,a[i]):
            matrix[i][k]=1
            idx+=1
    return matrix

def calc_RUE(matrix, software_count, service_avail, server_avail, r_add, H):
    initial_redundancy = [1] * software_count  # 冗長化度合いを1で初期化
    redundancy_cost_efficiency = []
    software_availability = [calc_software_av(matrix[j], service_avail, server_avail) for j in range(software_count)]
    system_avail = np.prod([1 - (1 - sa) ** int(r) for sa, r in zip(software_availability, initial_redundancy)])
    total_servers = sum(initial_redundancy[j] * ((r_add * (np.sum(matrix[j]) - 1)) + 1) for j in range(software_count))

    for i in range(software_count):
        initial_redundancy[i] = 2  # 一つのソフトウェアの冗長化度合いを2に変更
        total_servers_red = sum(initial_redundancy[j] * ((r_add * (sum(matrix[j]) - 1)) + 1) for j in range(software_count))
        
        if total_servers_red <= H:
            # 冗長化後のシステム全体の可用性を計算
            system_avail_red = np.prod([1 - (1 - sa) ** int(r) for sa, r in zip(software_availability, initial_redundancy)])
            
            # 冗長化コスト効率を計算し、リストに追加
            redundancy_cost_efficiency.append((system_avail_red - system_avail) / (total_servers_red - total_servers))
        else:
            redundancy_cost_efficiency.append(0)

        initial_redundancy[i] = 1  # 再度冗長化度合いを1にリセット

    avg_efficiency = np.mean(redundancy_cost_efficiency)

    return avg_efficiency

def greedy_search(matrix, software_count, service_avail, server_avail, r_add, H,service):
    list = []
    best_matrix = matrix.copy()
    best_RUE = calc_RUE(matrix, software_count, service_avail, server_avail, r_add, H)

    # 1つのソフトウェアの冗長化度合いを変更するか、ソフトウェア数を変更して探索
    for k in range(GENERATION):
        for j in range(software_count):
            new_matrix = matrix.copy()
            new_matrix[:, j] = np.roll(new_matrix[:, j], 1)  # 一つのサービス実装を変更
            new_RUE = calc_RUE(new_matrix, software_count, service_avail, server_avail, r_add, H)
            max_RUE = new_RUE
            
            if software_count <= 9: #sw数を増やす
                new_sw_p_matrix = make_matrix(service,software_count+1)
                new_RUE_p = calc_RUE(new_sw_p_matrix, software_count, service_avail, server_avail, r_add, H)
                max_RUE = max(new_RUE,new_RUE_p)
                
            elif software_count >= 2: #sw数を減らす
                new_sw_n_matrix = make_matrix(service,software_count-1)
                new_RUE_n = calc_RUE(new_sw_n_matrix, software_count, service_avail, server_avail, r_add, H)
                max_RUE = max(new_RUE,new_RUE_p,new_RUE_n)
                
            if max_RUE > best_RUE:
                if max_RUE == new_RUE:
                    best_RUE = new_RUE
                    best_matrix = new_matrix
                elif max_RUE == new_RUE_p:
                    best_RUE = max_RUE
                    best_matrix = new_sw_p_matrix
                elif max_RUE == new_RUE_n:
                    best_RUE = max_RUE
                    best_matrix = new_sw_n_matrix
            else:
                best_RUE = max_RUE
        list.append(best_RUE)
    
    return best_matrix, software_count, best_RUE, list

def multi_start_greedy(r_add, service_avail, server_avail, H, num_service, num_starts=10):
    best_global_matrix = None
    best_global_RUE = -np.inf
    best_global_software_count = 0
    RUE_list = []
    x_gene = [i for i in range(1,GENERATION+1)]
    service = [i for i in range(1,num_service)]

    for _ in range(num_starts):
        # 初期化：r_addに応じてソフトウェア数を少なくする
        software_count = np.random.randint(1, num_service)##
        matrix = np.zeros((software_count, num_service), dtype=int)
        if software_count == 1:
            matrix = np.ones((software_count,num_service),dtype = int)
            continue
        else:
            matrix = make_matrix(service,software_count)
        # 貪欲法で探索を実行
        best_matrix, best_software_count, best_RUE, RUE_each_list = greedy_search(matrix, software_count, service_avail, server_avail, r_add, H,service)
        RUE_list.extend(RUE_each_list)
        
        # グローバルな最良結果を更新
        if best_RUE > best_global_RUE:
            best_global_RUE = best_RUE
            best_global_matrix = best_matrix
            best_global_software_count = best_software_count

        plt.plot(x_gene,RUE_each_list)
    return best_global_matrix, best_global_software_count, best_global_RUE

# 使用例
r_add = 0.5  # 例としてr_add値
num_service = 10 #サービス数
service_avail = [0.99]*num_service  # サービス可用性の例
server_avail = 0.99  # サーバー可用性の例
H = 20  # 最大サーバー制約の例

GENERATION = 20

fig, ax = plt.subplots(figsize=(12, 8))
best_matrix, best_software_count, best_RUE = multi_start_greedy(r_add, service_avail, server_avail, H, num_service)

plt.xlabel("Generation")
plt.ylabel("RCE")
plt.show()

print(f"Best Matrix:\n{best_matrix}")
print(f"Best Software Count: {best_software_count}")
print(f"Best RCE: {best_RUE}")
