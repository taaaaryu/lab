import numpy as np

#制限なしナップサック問題を貪欲法で解く

software = 3
sw_avails = [0.99, 0.95, 0.98]
sw_resources = [6,4,5]
Resource = 25
server_avails = [0.99]*software
Max_Red = 4

def calc_system(redundancy, avails):
    system_avail = np.prod([1 - (1 - avail) ** int(red) for red, avail in zip(redundancy, avails)])
    return system_avail

sw_red = [1 for j in range(software)]

sum(sw_resources * sw_red) <= Resource

def solve():
    for i in range(software):
        for j in range(1, Resource):
            if res < sw_resources[i]:
                sw_red[i + 1][j] = sw_red[i][j]
            else:
                sw_red[i + 1][j] = max()
                
for i in range(n):
    for j in range(W + 1):
        if j < w[i]:
            dp[i + 1][j] = dp[i][j]
        else:
            dp[i + 1][j] = max(dp[i][j], dp[i + 1][j - w[i]] + v[i])

print(dp[n][W])
            

