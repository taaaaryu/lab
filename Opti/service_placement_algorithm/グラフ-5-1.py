import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

# Load the two datasets
file_proposed = 'results.csv'
file_exhaustive = 'results.csv'

# Read the datasets into DataFrames
df_proposed = pd.read_csv(file_proposed)
df_exhaustive = pd.read_csv(file_exhaustive)

# Merge the data based on common columns ('num_service', 'r_add')
merged_data = pd.merge(df_proposed, df_exhaustive, on=['num_service', 'r_add'], suffixes=('_proposed', '_exhaustive'))

# Plot settings
r_add_values = sorted(merged_data['r_add'].unique())
colors = ['b', 'g', 'r', 'c', 'm', 'y']  # Define a consistent color palette for r_add values

# Function to plot comparisons with error bars and consistent colors
def plot_comparison(metric_avg_prop, metric_max_prop, metric_min_prop,
                    metric_avg_exh, metric_max_exh, metric_min_exh,
                    ylabel, title,num):
    plt.figure(figsize=(12, 8))
    for i, r_add in enumerate(r_add_values):
        data = merged_data[merged_data['r_add'] == r_add]
        color = colors[i % len(colors)]  # Use the same color for the same r_add
        
        # Proposed Method
        y_prop = data[metric_avg_prop]
        yerr_prop = [y_prop - data[metric_min_prop], data[metric_max_prop] - y_prop]
        plt.errorbar(
            data['num_service'],
            y_prop,
            yerr=yerr_prop,
            fmt='o-',
            color=color,
            label='提案手法 $\mathrm{r_{add}}$='+f'{r_add}'
        )
        
        # Exhaustive Search
        y_exh = data[metric_avg_exh]
        yerr_exh = [y_exh - data[metric_min_exh], data[metric_max_exh] - y_exh]
        plt.errorbar(
            data['num_service'],
            y_exh,
            yerr=yerr_exh,
            fmt='s--',
            color=color,
            label='全探索手法 $\mathrm{r_{add}}$='+f'{r_add}'
        )
    
    plt.xlabel('サービス数', fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"5-1-{num}.png")
    plt.savefig(f'5-1-{num}.eps', bbox_inches="tight", pad_inches=0.05)

# Plot execution time comparison with error bars and consistent colors
plot_comparison(
    'time_avg_proposed', 'time_max_proposed', 'time_min_proposed',
    'time_avg_exhaustive', 'time_max_exhaustive', 'time_min_exhaustive',
    '実行時間(秒)', 'Execution Time Comparison: Proposed vs Exhaustive Search',1
)

# Plot unavailability comparison with error bars and consistent colors
plot_comparison(
    'unav_avg_proposed', 'unav_max_proposed', 'unav_min_proposed',
    'unav_avg_exhaustive', 'unav_max_exhaustive', 'unav_min_exhaustive',
    '非可用性', 'Unavailability Comparison: Proposed vs Exhaustive Search',2
)
