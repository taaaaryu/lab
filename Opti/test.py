import time
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations, chain
from mpl_toolkits.mplot3d import Axes3D
import os

# ディレクトリを作成して、画像を保存する準備をする
output_dir = 'output_graphs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 定数
n_values = [10]  # サービス数の異なる値
h_add_values = [0.5, 1.0, 1.5]  # h_addの異なる値
redundancy = 5  # 最大の冗長化数

for n in n_values:
    for h_add in h_add_values:
        services = [i for i in range(1, n + 1)]
        service_avail = [0.7, 0.8 ,0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
        server_avail = 0.9
        H = 20  # 全体のリソース
        
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
        num_comb = len(all_combinations)

        # 冗長度合いを計算し、システム非可用性を記録
        system_result = []

        for r in range(1, redundancy + 1):
            one_result = []
            for comb in all_combinations:
                software_availability = [calc_software_av(group) * server_avail for group in comb]
                system_unavail = 1 - np.prod([1 - (1 - sa) ** r for sa in software_availability])
                one_result.append(system_unavail)
            system_result.append(one_result)

        # サービスの組み合わせを表示するためにラベルを作成
        comb_labels = ['\n'.join([str(s) for s in comb]) for comb in all_combinations]
        comb_labels_sparse = [comb_labels[i] if i % (2**(n-3)-1) == 0 else '' for i in range(len(comb_labels))]

        # 各冗長化数における最小の可用性を示すソフトウェアの組み合わせを特定
        min_indices = [np.argmin(system_result[r-1]) for r in range(1, redundancy + 1)]

        # x軸を冗長化度、y軸をサービスの組み合わせ、z軸をシステムの非可用性としてプロット
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        X = np.arange(1, num_comb + 1)
        Y = np.arange(1, redundancy + 1)
        X, Y = np.meshgrid(X, Y)
        Z = np.array(system_result)

        ax.plot_surface(Y, X, Z, cmap='viridis')

        # Y軸の目盛りをサービスの組み合わせに変更
        ax.set_xticks(np.arange(1, redundancy + 1))
        ax.set_yticks(np.arange(1, num_comb + 1))
        ax.set_yticklabels(comb_labels_sparse, fontsize=8, rotation=0, va='center', ha='right', rotation_mode='anchor')

        # 各冗長化数におけるすべてのサービス組み合わせのシステム可用性を黒線で追加
        for y in range(1, redundancy + 1):
            ax.plot(np.full(num_comb, y), np.arange(1, num_comb + 1), Z[y-1, :], color='k',  linestyle='-', linewidth=1, zorder=10)

        # 各冗長化数における最小の可用性を示すソフトウェアの組み合わせを色を変えて表示
        for r in range(1, redundancy + 1):
            min_index = min_indices[r-1]
            ax.scatter(r, min_index+1, Z[r-1, min_index], color='r', s=100, zorder=11)

        ax.set_xlabel('Redundancy Level')
        ax.set_ylabel('Service Combinations')
        ax.set_zlabel('System Unavailability (log scale)')
        ax.set_zscale('log')
        ax.set_title(f'num of service={n}, server_availability = {server_avail}, h_add = {h_add}, resource = {H}')

        ax.view_init(elev=20, azim=30)  # elev: 上下の角度, azim: 水平の角度

        # 画像を保存
        plt.savefig(f'{output_dir}/graph_n{n}_hadd{h_add}.png')
        plt.close()

print("画像の保存が完了しました。")
