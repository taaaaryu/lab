import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

# Load the dataset
file_proposed = '/Users/taaaaryu/Desktop/研究室/lab/Opti/results-big.csv'

# Read the dataset into a DataFrame
df_proposed = pd.read_csv(file_proposed)

# Plot settings
r_add_values = sorted(df_proposed['r_add'].unique())
colors = ['b', 'g', 'r', 'c', 'm', 'y']  # Define a consistent color palette for r_add values

# Function to plot the proposed method with error bars and consistent colors
def plot_proposed(metric_avg, metric_max, metric_min, ylabel, title, num):
    plt.figure(figsize=(12, 8))
    for i, r_add in enumerate(r_add_values):
        data = df_proposed[df_proposed['r_add'] == r_add]
        color = colors[i % len(colors)]  # Use the same color for the same r_add
        
        # Proposed Method
        y = data[metric_avg]
        yerr = [y - data[metric_min], data[metric_max] - y]
        plt.errorbar(
            data['num_service'],
            y,
            yerr=yerr,
            fmt='o-',
            color=color,
            label='提案手法 $\mathrm{r_{add}}$='+f'{r_add}'
        )
    
    plt.xlabel('サービス数', fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    if ylabel == '非可用性':
        plt.yscale('log')
    plt.legend()
    plt.grid(True)
    #plt.title(title, fontsize=18)
    plt.savefig(f"5-2-{num}.png")
    plt.savefig(f'5-2-{num}.eps', bbox_inches="tight", pad_inches=0.05)

# Plot execution time for the proposed method with error bars
plot_proposed(
    'time_avg', 'time_max', 'time_min',
    '実行時間(秒)', '提案手法の実行時間', 1
)

# Plot unavailability for the proposed method with error bars
plot_proposed(
    'unav_avg', 'unav_max', 'unav_min',
    '非可用性', '提案手法の非可用性', 2
)
