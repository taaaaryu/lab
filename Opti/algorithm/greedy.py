import numpy as np
import random

def calc_software_av(services_in_sw, service_avail,server_avail):
    # 各SWの可用性を計算
    non_sero_idx = np.nonzero(services_in_sw)
    sw_avail=1
    non_sero_idx= [item.tolist() for item in non_sero_idx]

    for k in non_sero_idx[0]:
        sw_avail *= service_avail[k]*services_in_sw[k]
    return sw_avail*server_avail

def calc_RUE(matrix, software_count, service_avail, server_avail, r_add, H):
    initial_redundancy = [1] * software_count  # 冗長化度合いを1で初期化
    redundancy_cost_efficiency = []
    software_availability = [calc_software_av(matrix[j], service_avail, server_avail) for j in range(software_count)]
    #print(software_availability)
    system_avail = np.prod([1 - (1 - sa) ** int(r) for sa, r in zip(software_availability, initial_redundancy)])
    total_servers = sum(initial_redundancy[j] * ((r_add * (np.sum(matrix[:, j]) - 1)) + 1) for j in range(software_count))

    for i in range(software_count):
        initial_redundancy[i] = 2  # 一つのソフトウェアの冗長化度合いを2に変更
        total_servers_red = sum(initial_redundancy[j] * ((r_add * (np.sum(matrix[:, j]) - 1)) + 1) for j in range(software_count))
        
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

def greedy_search(matrix, software_count, service_avail, server_avail, r_add, H):
    best_matrix = matrix.copy()
    best_RUE = calc_RUE(matrix, software_count, service_avail, server_avail, r_add, H)

    # 1つのソフトウェアの冗長化度合いを変更するか、ソフトウェア数を変更して探索
    for i in range(software_count):
        for j in range(software_count):
            new_matrix = matrix.copy()
            new_matrix[:, j] = np.roll(new_matrix[:, j], 1)  # 一つのサービス実装を変更
            new_RUE = calc_RUE(new_matrix, software_count, service_avail, server_avail, r_add, H)
            if new_RUE > best_RUE:
                best_RUE = new_RUE
                best_matrix = new_matrix

    return best_matrix, software_count, best_RUE

def multi_start_greedy(r_add, service_avail, server_avail, H, num_service, num_starts=10):
    best_global_matrix = None
    best_global_RUE = -np.inf
    best_global_software_count = 0
    service = [i for i in range(1,num_service)]

    for _ in range(num_starts):
        # 初期化：r_addに応じてソフトウェア数を少なくする
        software_count = np.random.randint(1, num_service)##
        matrix = np.zeros((software_count, num_service), dtype=int)
        if software_count == 1:
            continue
        else:
            a = random.sample(service,software_count-1)
            a.append(num_service)
            a.sort()
            idx = 0
            for i in range(software_count):
                for k in range(idx,a[i]):
                    matrix[i][k]=1
                    idx+=1
        print(matrix)
        # 貪欲法で探索を実行
        best_matrix, best_software_count, best_RUE = greedy_search(matrix, software_count, service_avail, server_avail, r_add, H)
        
        # グローバルな最良結果を更新
        if best_RUE > best_global_RUE:
            best_global_RUE = best_RUE
            best_global_matrix = best_matrix
            best_global_software_count = best_software_count

    return best_global_matrix, best_global_software_count, best_global_RUE

# 使用例
r_add = 0.5  # 例としてr_add値
num_service = 10 #サービス数
service_avail = [0.99]*num_service  # サービス可用性の例
server_avail = 0.99  # サーバー可用性の例
H = 20  # 最大サーバー制約の例

best_matrix, best_software_count, best_RUE = multi_start_greedy(r_add, service_avail, server_avail, H, num_service)
print(f"Best Matrix:\n{best_matrix}")
print(f"Best Software Count: {best_software_count}")
print(f"Best RCE: {best_RUE}")
