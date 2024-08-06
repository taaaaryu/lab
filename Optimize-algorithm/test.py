from datetime import datetime
import random
from decimal import Decimal
import numpy as np
from itertools import zip_longest

class Chronosome:
    genom = None
    evaluation = None
    
    def __init__(self,genom,evaluation):
        self.genom = genom
        self.evaluation = evaluation
    
    def getGenom(self):
        return self.genom
    
    def getEvaluation(self):
        return self.evaluation
    
    def setGenom(self,genom_list):
        self.genom = genom_list
    
    def setEvaluation(self,evaluation):
        self.evaluation = evaluation

def create_Chromosome(length):
    '''genom_list=[[Aさんの出勤],[Bさんの出勤],[C...],[D...]]'''
    days_list = []
    genom_list = []

    for i in range(int(GENOM_LENGTH/DAY)):
        for j in range(DAY):
            days_list.append(float(random.randint(0,1)))
        genom_list.append(days_list)
        days_list = []
    return Chronosome(genom_list,0)

def evaluation(Chronosome):
    """評価関数。制約式のペナルティーを適応度とする
    :param Chromosome: 評価を行うChromosomeClass
    :return: 評価処理をしたChromosomeClassを返す
    """
    fitness = constraints(Chronosome)
    return fitness

def constraints(Chronosome):
    penalty = 0.0

    genom_arr = np.array(Chronosome.getGenom())
    for i in range(genom_arr.shape[0]):
        employee = genom_arr[i]
        if sum(employee) > MAX_SHIFT:
            penalty += 50*abs(sum(employee)-MAX_SHIFT)
    for i in range(genom_arr.shape[1]):
        if sum([shift[i] for shift in genom_arr] != SHIFT[i][1]):
            penalty += 10*abs(sum(shift[i] for shift in genom_arr)-SHIFT[i][1])
    return penalty

def roulette_select(Chronosome, choise_num):
    fitness_arr = np.array([float(genom.evaluation) for genom in Chronosome])
    fitness_norm = np.random.choise(np.arange(len(Chronosome)),size = choice_num,p = fitness_arr/sum(fitness_arr))
    result = [Chronosome[i] for i in fitness_norm]
    return result

def tornament_select(Chronosome):
    fitness_arr = [float(genom.evaluation) for genom in Chronosome]
    next_gene_arr = []
    for i in range(choice_num):
        [idx_choice1,idx_choice2] = np.random.randint(MAX_GENOM_LIST,size=2)
        if fitness_arr[idx_choice1]>fitness_arr[idx_choice2]:
            next_gene_arr.append(Chronosome[idx_choice1])
        else:
            next_gene_arr.append(Chronosome[idx_choice2])
    return np.array(next_gene_arr)

def uniform_crossover(Chronosome1,Chronosome2):
    #一様交差
    genom_list = []
    one = Chronosome1.getGenom()
    second = Chronosome2.getGenom()

    #crossover
    for i in range(len(one)):
        if np.random() < 0.5:
            genom_list.append(one[i])
        else:
            genom_list.append(second[i])
            
    return [Chromosome(genom_list, 0)]

def mutation(Chronosome,genom_mutation, individual_mutation):
    Chronosome_list = []
    for genom in Chronosome:
        if genom_mutation > (random.randint(0,100)/100):
            genom_list =[]
            for k in genom.getGenom():
                ga_list = []
                if genom_mutation> (random.randint(0,100)/100):
                     for j in range(len(k)):
                        ga_list.append(float(random.randint(0.0, 1.0)))
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
    next_generation_geno = sorted(Chromosome, reverse=True, key=lambda u: u.evaluation)
    # 追加するエリート集団と子孫集団の合計分を取り除く
    for i in range(0, len(Chromosome_elite) + len(Chromosome_progeny)):
        next_generation_geno.pop(0)
    # エリート集団と子孫集団を次世代集団を次世代へ追加
    next_generation_geno.extend(Chromosome_elite)
    next_generation_geno.extend(Chromosome_progeny)
    return next_generation_geno

# シフトの定義（シフト, 必要人数）
SHIFT = [('Mon',2), ('Tue',2), ('Wed',2), ('Thu',1), ('Fri',2), ('Sat',3), ('Sun',3)]
# シフト
DAY = len(SHIFT)
# 従業員数
PEOPLE = 5
# 各従業員の出勤最大日数
MAX_SHIFT = 3
# 遺伝子情報の長さ
GENOM_LENGTH = PEOPLE * DAY
# 遺伝子集団の大きさ
MAX_GENOM_LIST = 300
# 遺伝子選択数
SELECT_GENOM = 40
# 個体突然変異確率
INDIVIDUAL_MUTATION = 0.1
# 遺伝子突然変異確率
GENOM_MUTATION = 0.1
# 繰り返す世代数
MAX_GENERATION = 40
# 繰り返しをやめる評価値の閾値
THRESSHOLD = 0


# 一番最初の現行世代個体集団を生成
current_generation_individual_group = []
for i in range(MAX_GENOM_LIST):
    current_generation_individual_group.append(create_Chromosome(GENOM_LENGTH))

for count_ in range(1, MAX_GENERATION + 1):
    # 現行世代個体集団の遺伝子を評価し、ChromosomeClassに代入
    for i in range(MAX_GENOM_LIST):
        evaluation_result = evaluation(current_generation_individual_group[i])
        current_generation_individual_group[i].setEvaluation(evaluation_result)
    # エリート個体を選択
    elite_genes = elite_select(current_generation_individual_group,SELECT_GENOM)
    # エリート遺伝子を交叉させ、リストに格納
    progeny_gene = []
    for i in range(0, SELECT_GENOM):
        progeny_gene.extend(crossover(elite_genes[i - 1], elite_genes[i]))
    # 次世代個体集団を現行世代、エリート集団、子孫集団から作成
    next_generation_individual_group = next_generation_gene_create(current_generation_individual_group,
                                                                   elite_genes, progeny_gene)
    # 次世代個体集団全ての個体に突然変異を施す
    next_generation_individual_group = mutation(next_generation_individual_group,INDIVIDUAL_MUTATION,GENOM_MUTATION)

    # 1世代の進化的計算終了

    # 各個体適用度を配列化
    fits = [i.getEvaluation() for i in current_generation_individual_group]

    # 進化結果を評価
    min_ = min(fits)
    max_ = max(fits)
    avg_ = Decimal(sum(fits)) / Decimal(len(fits))

    # 現行世代の進化結果を出力します
    print(datetime.now(),
          f'世代数 : {count_}  ',
          f'Min : {min_:.3f} ',
          f'Max : {max_:.3f}  ',
          f'Avg : {avg_:.3f}  '
         )
    # 現行世代と次世代を入れ替える
    current_generation_individual_group = next_generation_individual_group
    # 適応度が閾値に達したら終了
    if THRESSHOLD >= min_:
        print('optimal')
        print(datetime.now(),
          f'世代数 : {count_}  ',
          f'Min : {min_:.3f} ',
          f'Max : {max_:.3f}  ',
          f'Avg : {avg_:.3f}  '
         )
        break
# 最最良個体結果出力
print(f'最良個体情報:{elite_genes[0].getGenom()}')