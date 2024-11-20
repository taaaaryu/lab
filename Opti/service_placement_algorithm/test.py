import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

# Load the two datasets
file_proposed = '/Users/taaaaryu/Desktop/研究室/lab/Opti/results_提案手法.csv'
file_exhaustive = '/Users/taaaaryu/Desktop/研究室/lab/Opti/results_全探索.csv'

# Read the datasets into DataFrames
df_proposed = pd.read_csv(file_proposed)
df_exhaustive = pd.read_csv(file_exhaustive)

# Merge the data based on common columns ('num_service', 'r_add')
merged_data = pd.merge(df_proposed, df_exhaustive, on=['num_service', 'r_add'], suffixes=('_proposed', '_exhaustive'))

# Plot settings
r_add_values = sorted(merged_data['r_add'].unique())
colors = ['b', 'g', 'r', 'c', 'm', 'y']  # Define a consistent color palette for r_add values

# Function to create separate plots
def plot_separate(metric_avg_time_prop, metric_max_time_prop, metric_min_time_prop,
                  metric_avg_time_exh, metric_max_time_exh, metric_min_time_exh,
                  metric_avg_unav_prop, metric_max_unav_prop, metric_min_unav_prop,
                  metric_avg_unav_exh, metric_max_unav_exh, metric_min_unav_exh):
    # Execution Time Comparison
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.set_title("実行時間の比較 (提案手法 vs 全探索)", fontsize=18)
    for i, r_add in enumerate(r_add_values):
        data = merged_data[merged_data['r_add'] == r_add]
        color = colors[i % len(colors)]
        
        # Proposed Method
        y_prop = data[metric_avg_time_prop]
        yerr_prop = [y_prop - data[metric_min_time_prop], data[metric_max_time_prop] - y_prop]
        ax1.errorbar(
            data['num_service'], y_prop, yerr=yerr_prop, fmt='o-', color=color,
            label=f'提案手法 $\mathrm{{r_{{add}}}}$={r_add}'
        )
        
        # Exhaustive Search
        y_exh = data[metric_avg_time_exh]
        yerr_exh = [y_exh - data[metric_min_time_exh], data[metric_max_time_exh] - y_exh]
        ax1.errorbar(
            data['num_service'], y_exh, yerr=yerr_exh, fmt='s--', color=color,
            label=f'全探索手法 $\mathrm{{r_{{add}}}}$={r_add}'
        )
    ax1.set_xlabel("サービス数", fontsize=14)
    ax1.set_ylabel("実行時間 (秒)", fontsize=14)
    ax1.set_yscale('log')
    ax1.grid(True)
    ax1.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("execution_time_comparison.png")
    plt.savefig("execution_time_comparison.eps", bbox_inches="tight", pad_inches=0.05)
    plt.show()

    # Unavailability Comparison
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.set_title("非可用性の比較 (提案手法 vs 全探索)", fontsize=18)
    for i, r_add in enumerate(r_add_values):
        data = merged_data[merged_data['r_add'] == r_add]
        color = colors[i % len(colors)]
        
        # Proposed Method
        y_prop = data[metric_avg_unav_prop]
        yerr_prop = [y_prop - data[metric_min_unav_prop], data[metric_max_unav_prop] - y_prop]
        ax2.errorbar(
            data['num_service'], y_prop, yerr=yerr_prop, fmt='o-', color=color,
            label=f'提案手法 $\mathrm{{r_{{add}}}}$={r_add}'
        )
        
       # Exhaustive Search
        y_exh = data[metric_avg_unav_exh]
        yerr_exh = [y_exh - data[metric_min_unav_exh], data[metric_max_unav_exh] - y_exh]
        ax2.errorbar(
            data['num_service'], y_exh, yerr=yerr_exh, fmt='s--', color=color,
            label=f'全探索手法 $\mathrm{{r_{{add}}}}$={r_add}'
        )
    ax2.set_xlabel("サービス数", fontsize=14)
    ax2.set_ylabel("非可用性", fontsize=14)
    ax2.set_yscale('log')
    ax2.grid(True)
    ax2.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("unavailability_comparison.png")
    plt.savefig("unavailability_comparison.eps", bbox_inches="tight", pad_inches=0.05)
    plt.show()

# Call the function with the appropriate metrics
plot_separate(
    'time_avg_proposed', 'time_max_proposed', 'time_min_proposed',
    'time_avg_exhaustive', 'time_max_exhaustive', 'time_min_exhaustive',
    'unav_avg_proposed', 'unav_max_proposed', 'unav_min_proposed',
    'unav_avg_exhaustive', 'unav_max_exhaustive', 'unav_min_exhaustive'
)