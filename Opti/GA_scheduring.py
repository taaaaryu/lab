from datetime import datetime
import random
from decimal import Decimal
import numpy as np
from itertools import zip_longest
import matplotlib.pyplot as plt

# 染色体の定義
class Chromosome:

    genom = None
    evaluation = None

    def __init__(self, genom, evaluation):
        self.genom = genom
        self.evaluation = evaluation

    def getGenom(self):
        return self.genom

    def getEvaluation(self):
        return self.evaluation

    def setGenom(self, genom_list):
        self.genom = genom_list

    def setEvaluation(self, evaluation):
        self.evaluation = evaluation
        
# ソフトウェアの可用性を計算する関数
def calc_software_av(service_comb, service_avail):
    sw_avails = []
    sv_count = 0
    #
    for j in service_comb:
        Software_avail = 1
        j += sv_count
        while sv_count < j:
            Software_avail *= service_avail[sv_count]
            sv_count += 1
        sw_avails.append(Software_avail*SERVER_AVAIL)
    return sw_avails

# それぞれのソフトウェアのリソースを計算する関数
def calc_software_resource(service_comb, r_add):
    sw_resources = []
    for i in service_comb:
        each_resource = 1 + r_add * (i - 1)
        sw_resources.append(each_resource)
    return sw_resources

def create_Chromosome(length):
    """
    引数で指定された桁のランダムな遺伝子情報を生成、格納したChromosomeClassで返す。
    :param length: 遺伝子情報長
    :return: 生成した個体集団ChromosomeClass
    """
    sw_list = []
    genom_list = []
    
    for i in range(int(GENOM_LENGTH/SOFTWARE)):
        sw_list = [0]*(SOFTWARE)
        idx = random.randint(0,SOFTWARE - 1)
        sw_list[idx] = 1
        genom_list.append(sw_list)
        sw_list = []
    return Chromosome(genom_list, 0)

def calc_system(redundancy):
    system_avail = np.prod([1 - (1 - avail) ** int(red) for red, avail in zip(redundancy, SW_AVAILS)])
    return system_avail

def evaluation(Chromosome):
    """評価関数。制約式のペナルティーを適応度とする
    :param Chromosome: 評価を行うChromosomeClass
    :return: 評価処理をしたChromosomeClassを返す
    """
    redundancy = np.sum(Chromosome.getGenom(), axis=0)
    fitness = calc_system(redundancy) - constraints(Chromosome)
    return fitness

def constraints(Chromosome):
    """制約関数。制約が満たされない場合、ペナルティーを付与する。
    :param Chromosome: 評価を行うChromosomeClass
    :return: penalty 
    """
    global penalty
    penalty = 0.0
    # 多次元配列から行列に変換
    genom_arr = np.array(Chromosome.getGenom())
    genom_raw_sum = np.sum(genom_arr,axis=0)
    
    # リソースを超えないようにする
    if np.dot(genom_raw_sum, SW_RESOURCES) > RESOURCE:
        penalty += 100.0
    # 各サーバに１つ以下のソフトウェアがホストされる制約(0はホストされない)
    for i in range(genom_arr.shape[0]):
        server = genom_arr[i]
        if sum(server) > MAX_PLACE:
            penalty += 50.0 * abs(sum(server) - MAX_PLACE)
    # 各ソフトウェアの冗長化数が最大で4となる
    for j in range(SOFTWARE):
        if genom_raw_sum[j] > MAX_REDUNDANCY:
            penalty += 30.0 * abs(genom_raw_sum[j] - MAX_REDUNDANCY)

    return penalty

def elite_select(Chromosome, elite_length):
    """選択関数です。エリート選択
    評価が高い順番にソートを行った後、一定以上の染色体を選択
    :param Chromosome: 選択を行うChromosomeClassの配列
    :param elite_length: 選択する染色体数
    :return: 選択処理をした一定のエリート、ChromosomeClassを返す
    """
    # 現行世代個体集団の評価を低い順番にソートする
    sort_result = sorted(Chromosome, reverse=True, key=lambda u: u.evaluation)
    # 一定の上位を抽出する
    result = [sort_result.pop(0) for i in range(elite_length)]
    return result


def roulette_select(Chromosome, choice_num):
    """選択関数です。ルーレット選択
    適応度に応じた重み付きでランダムに選択
    :param Chromosome: 選択を行うChromosomeClassの配列
    :param elite_length: 選択する染色体数
    :return: 選択処理をした染色体、ChromosomeClassを返す
    """
    # 適応度を配列化
    fitness_arr = np.array([float(genom.evaluation) for genom in Chromosome])
    
    idx = np.random.choice(np.arange(len(Chromosome)), size=choice_num, p=fitness_arr/sum(fitness_arr))
    result = [Chromosome[i] for i in idx]
    return result

def tournament_select(Chromosome, choice_num):
    """選択関数です。トーナメント選択
    評価が高い順番にソートを行った後、一定以上の染色体を選択
    :param Chromosome: 選択を行うChromosomeClassの配列
    :param elite_length: 選択する染色体数
    :return: 選択処理をした遺伝子、ChromosomeClassを返す
    """
    # 適応度を配列化
    fitness_arr = [float(genom.evaluation) for genom in Chromosome]
    print(max(fitness_arr))
    next_gene_arr = []
    for i in range(choice_num):
        [idx_chosen1, idx_chosen2] = np.random.randint(MAX_GENOM_LIST, size=2)
        if fitness_arr[idx_chosen1] > fitness_arr[idx_chosen2]:
            next_gene_arr.append(Chromosome[idx_chosen1])
        else:
            next_gene_arr.append(Chromosome[idx_chosen2])

    return np.array(next_gene_arr)


def crossover(Chromosome_one, Chromosome_second):
    """交叉関数。二点交叉
    :param Chromosome: 交叉させるChromosomeClassの配列
    :param Chromosome_one: 一つ目の個体
    :param Chromosome_second: 二つ目の個体
    :return: 二つの子孫ChromosomeClassを格納したリスト返す
    """
    # 子孫を格納するリストを生成
    genom_list = []
    # 入れ替えるサーバを選択
    idx = random.randint(0,SERVER-1)
    # 遺伝子を取り出し
    one = Chromosome_one.getGenom()
    second = Chromosome_second.getGenom()
    # 交叉
    copy_one = one.copy()
    second[idx] = copy_one[idx]
 
    # ChromosomeClassインスタンスを生成して子孫をリストに格納
    genom_list.append(Chromosome(one, 0))
    genom_list.append(Chromosome(second, 0))
    return genom_list

def uniform_crossover(Chromosome_one, Chromosome_second):
    """交叉関数。一様交叉。
    :param Chromosome: 交叉させるChromosomeClassの配列
    :param Chromosome_one: 一つ目の個体
    :param Chromosome_second: 二つ目の個体
    :return: 二つの子孫ChromosomeClassを格納したリスト返す
    """
    # 子孫を格納するリストを生成
    genom_list = []
    # 遺伝子を取り出す
    one = Chromosome_one.getGenom()
    print(one)
    second = Chromosome_second.getGenom()
    # 交叉(サーバを入れ替える)
    for i in range(SERVER):
        if np.random.rand() < CROSSOVER_PRO:
            second[i] = one[i]
            genom_list.append(one[i])
        else:
            genom_list.append(second[i])
    return [Chromosome(genom_list, 0)]

def mutation(Chromosome, individual_mutation, genom_mutation):
    """突然変異関数。
    :param Chromosome: 突然変異をさせるChromosomeClass
    :param individual_mutation: 固定に対する突然変異確率
    :param Chromosome_mutation: 遺伝子一つ一つに対する突然変異確率
    :return: 突然変異処理をしたgenomClassを返す"""
    Chromosome_list = []
    for genom in Chromosome:
        # 個体に対して一定の確率で突然変異が起きる
        if individual_mutation > (random.randint(0, 100) / Decimal(100)):
            genom_list = []
            for i_ in genom.getGenom():
                ga_list = []
                # 個体の遺伝子情報一つ一つに対して突然変異が起こる
                if genom_mutation > (random.randint(0, 100) / Decimal(100)):
                    for j in range(len(i_)):
                        ga_list.append(float(random.randint(0.0,1.0)))
                    genom_list.append(ga_list)
                else:
                    genom_list.append(i_)
            genom.setGenom(genom_list)
            Chromosome_list.append(genom)
        else:
            Chromosome_list.append(genom)
    return Chromosome_list



def next_generation_gene_create(Chromosome, Chromosome_elite, Chromosome_progeny):
    """
    世代交代処理
    :param Chromosome: 現行世代個体集団
    :param Chromosome_elite: 現行世代エリート集団
    :param Chromosome_progeny: 現行世代子孫集団
    :return: 次世代個体集団
    """
    # 現行世代個体集団の評価を高い順番にソート
    next_generation_geno = sorted(Chromosome, reverse=False, key=lambda u: u.evaluation)
    # 追加するエリート集団と子孫集団の合計分を取り除く
    for i in range(0, len(Chromosome_elite) + len(Chromosome_progeny)):
        next_generation_geno.pop(0)
    # エリート集団と子孫集団を次世代集団を次世代へ追加
    next_generation_geno.extend(Chromosome_elite)
    next_generation_geno.extend(Chromosome_progeny)
    return next_generation_geno


# サーバ数
SERVER = 20
# リソース
RESOURCE = 40
#1つのサーバにホストされるSW数
MAX_PLACE=1
#最大の冗長化数
MAX_REDUNDANCY = 4


#サービスの分割
SERVICE_COMBINATION = [2,1,2,2,3,2,4,2,1,1]
#サービスの可用性
SERVICE_AVAILS = [0.99]*sum(SERVICE_COMBINATION)
# ソフトウェア数
SOFTWARE = len(SERVICE_COMBINATION)
#ソフトウェアの可用性とそれぞれのソフトウェアが必要とするリソース
SERVER_AVAIL = 0.99
#ソフトウェアに複数のサービスが内包される際のコスト
R_ADD = 1


# 遺伝子情報の長さ
GENOM_LENGTH = SERVER*SOFTWARE
# 遺伝子集団の大きさ
MAX_GENOM_LIST = 300
# 遺伝子選択数
SELECT_GENOM = 50
#交叉の確率
#CROSSOVER_PRO = 0.5
# 個体突然変異確率
INDIVIDUAL_MUTATION = 0.05
# 遺伝子突然変異確率
GENOM_MUTATION = 0.05
# 繰り返す世代数
MAX_GENERATION = 50
# 繰り返しをやめる評価値の閾値
THRESSHOLD = 0.99999
#局所最適化対策、何回同じ解が続いたらリセットするか
LOCAL_OPTI = 3


#それぞれのソフトウェアの可用性とリソースを計算
SW_AVAILS = calc_software_av(SERVICE_COMBINATION,SERVICE_AVAILS)
SW_RESOURCES = calc_software_resource(SERVICE_COMBINATION, R_ADD)
print(SW_RESOURCES)

local_opti = []
Graph_Count = []
Graph_Result = []
Max_Redundancy = []

# 一番最初の現行世代個体集団を生成
current_generation_individual_group = []
for i in range(MAX_GENOM_LIST):
    current_generation_individual_group.append(create_Chromosome(GENOM_LENGTH))
#print(current_generation_individual_group)

for count_ in range(1, MAX_GENERATION + 1):
    # 現行世代個体集団の遺伝子を評価し、ChromosomeClassに代入
    for i in range(MAX_GENOM_LIST):
        evaluation_result = evaluation(current_generation_individual_group[i])
        current_generation_individual_group[i].setEvaluation(evaluation_result)
    # エリート個体を選択
    choice_genes = elite_select(current_generation_individual_group,SELECT_GENOM)
    # エリート遺伝子を交叉させ、リストに格納
    progeny_gene = []
    for i in range(0, SELECT_GENOM):
        progeny_gene.extend(crossover(choice_genes[i - 1], choice_genes[i]))
    # 次世代個体集団を現行世代、エリート集団、子孫集団から作成
    next_generation_individual_group = next_generation_gene_create(current_generation_individual_group,
                                                                   choice_genes, progeny_gene)
    # 次世代個体集団全ての個体に突然変異を施す
    next_generation_individual_group = mutation(next_generation_individual_group,INDIVIDUAL_MUTATION,GENOM_MUTATION)

    # 1世代の進化的計算終了

    # 各個体適用度を配列化
    fits = [i.getEvaluation() for i in current_generation_individual_group]

    # 進化結果を評価
    max_ = max(fits)
    Redundancy_result = np.sum(choice_genes[0].getGenom(),axis=0)
    # 現行世代の進化結果(最低の非可用性)を記録
    Graph_Count.append(count_)
    Max_Redundancy.append(Redundancy_result)
    Graph_Result.append(1 - max_)
    
    #局所最適化への対策として、ある回数解が同じになったら、リセット
    Local_Result = Graph_Result[-1*LOCAL_OPTI:]
    if all(elem == Local_Result[-1] for elem in Local_Result):
        current_generation_individual_group = []
        for i in range(MAX_GENOM_LIST):
            current_generation_individual_group.append(create_Chromosome(GENOM_LENGTH))
        #リセットの際、突然変異の起こる確率を上げる
        if INDIVIDUAL_MUTATION < 0.1:
            INDIVIDUAL_MUTATION += 0.01
        if GENOM_MUTATION < 0.1:
            GENOM_MUTATION += 0.01 
    else:
        # 現行世代と次世代を入れ替える
        current_generation_individual_group = next_generation_individual_group
    # 適応度が閾値に達したら終了
    if THRESSHOLD <= max_:
        print('optimal')
        break
# 最最良個体結果出力
#print(choice_genes[0].getGenom())
Unavail_min = min(Graph_Result)
result_idx = Graph_Result.index(Unavail_min)
print(f'最良個体情報:{Max_Redundancy[result_idx]}')
print(f'最大の可用性{1-Unavail_min}')
print(f'必要なリソース:{np.dot(Max_Redundancy[result_idx],SW_RESOURCES)}')

plt.subplots()
plt.plot(Graph_Count, Graph_Result)
plt.xlabel("generation")
plt.ylabel("system_Unavailability")
plt.yscale("log")

plt.show()