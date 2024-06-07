import gurobipy as gp
from gurobipy import GRB	

import time

# 重複あり、listの和がtargetとなる配列を出力する関数
def get_integral_value_combination(lst, target):
    def a(idx, l, r, t):
        if t == sum(l):
            r.append(l)
        elif t < sum(l):
            return
        for u in range(idx, len(lst)):
            a(u, l + [lst[u]], r, t)
        return r
    return a(0, [], [], target)

# 定数
c_sv = 4
c_nw = 3
num_service = 70
p = 0.985
q = 0.015
r = 0.2  # サービスの数が増えると可用性は何分の一になるか
a_sv = 0.999
a_nw = 0.999
a_sys = []

each_service = [i for i in range(1, num_service + 1)]

start_time = time.time()  # 処理開始時間

service_com = get_integral_value_combination(each_service, num_service)  # サービスの組み合わせリスト

I = range(len(service_com))

print("Number of combinations:", len(service_com))

for i in I:
    a_hold = 1
    for xi in service_com[i]:
        a_hold *= (p + q * (r ** (xi - 1))) * a_sv * a_nw
    a_sys.append(a_hold)

SLA_a_sys = 0.98

end_time = time.time()  # 処理終了時間
elapsed_time = end_time - start_time  # 経過時間

print("Elapsed time: {:.2f} seconds".format(elapsed_time))


print("---------------------------")

"""
m = gp.Model(name = "KS")  # 数理モデル

x = {}
for i in I:
	x[i] = m.addVar(vtype="B", name=f"service({i})")


#x = [m.add_var(var_type=BINARY) for i in I]
m.setObjective(sum(c_sv * x[i]* len(service_com[i]) + c_nw * x[i]* len(service_com[i]) for i in I), sense = gp.GRB.MINIMIZE)

m.addConstr(sum(x[i] for i in I) == 1)
m.addConstrs((x[i] == 1) >> (a_sys[i] * x[i] >= SLA_a_sys * x[i])) ,name = "avail")

m.optimize()



for i in range(len(service_com)):
     if x[i].X == 1:
         print(f'最適な組み合わせは{service_com[i]}で, その時のシステム可用性は{a_sys[i]}')

         """