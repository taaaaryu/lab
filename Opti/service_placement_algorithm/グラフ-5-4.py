import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib

# データ
service_count = [20, 40, 60, 80, 100]
r_add_08 = [4.2, 13.9, 21, 28.4, 37.7]
r_add_1 = [10.8, 18, 26.4, 33.6, 41.6]
r_add_12 = [20, 24.8, 34.1, 40.8, 49.2]

# 棒グラフの幅
bar_width = 0.25
index = np.arange(len(service_count))

# グラフの作成
plt.figure(figsize=(10, 6))
plt.bar(index - bar_width, r_add_08, width=bar_width, label='$\mathrm{r_{add}}$=0.8')
plt.bar(index, r_add_1, width=bar_width, label='$\mathrm{r_{add}}$=1.0')
plt.bar(index + bar_width, r_add_12, width=bar_width, label='$\mathrm{r_{add}}$=1.2')

# ラベル、タイトル、凡例
plt.xlabel('サービス数', fontsize=14)
plt.ylabel('平均ソフトウェア数', fontsize=14)
#plt.title('サービス数と平均ソフトウェア数の関係', fontsize=14)
plt.xticks(index, service_count)
plt.legend(fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# グラフ表示
plt.tight_layout()
plt.savefig("5-4.png")
plt.savefig('5-4.eps', bbox_inches="tight", pad_inches=0.05)