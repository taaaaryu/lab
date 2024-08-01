import numpy as np
import matplotlib.pyplot as plt


class Individual:
    '''各個体のクラス
        args: 個体の持つ遺伝子情報(np.array)'''
    def __init__(self, genom):
        self.genom = genom
        self.fitness = 0  # 個体の適応度(set_fitness関数で設定)
        self.set_fitness()

    def set_fitness(self):
        '''個体に対する目的関数(OneMax)の値をself.fitnessに代入'''
        self.fitness = self.genom.sum()

    def get_fitness(self):
        '''self.fitnessを出力'''
        return self.fitness

    def mutate(self):
        tmp = self.genom.copy()
        i = np.random.randint(0,len(self.genom) -1)
        tmp[i] = float(not self.genom[i])
        self.genom = tmp
        self.set_fitness()


def create_generation(POPURATIONS, GENOMS):
    '''初期世代の作成
    input:
        POPURATIONS: 1世代の個体数
        GENOMS: 遺伝子の長さ(n)
    return: 初期世代の個体クラスのリスト'''
    generation = []
    for i in range(POPURATIONS):
        individual = Individual(np.random.randint(0, 2, GENOMS)) # ランダムに0,1を並べた数列を作成
        generation.append(individual)
    return generation



def select_roulette(generation):
    selected = []
    weights = [ind.get_fitness() for ind in generation]
    norm_weights = [ind.get_fitness()/sum(weights) for ind in generation]
    selected = np.random.choice(generation, size=len(generation), p=norm_weights)
    return selected

def select_tournament(generation):
    '''選択の関数(トーナメント方式)'''
    selected = []
    for i in range(len(generation)):
        tournament = np.random.choice(generation, 3, replace=False)
        max_genom = max(tournament, key=Individual.get_fitness).genom.copy()
        selected.append(Individual(max_genom))
    return selected

def cross_two_point_copy(child1, child2):
    '''交叉の関数(二点交叉)
    input: 混ぜ合わせたい個体のペア
    output: 交叉後の個体のペア'''
    size = len(child1.genom)
    tmp1 = child1.genom.copy()
    tmp2 = child2.genom.copy()
    cx_point1 = np.random.randint(1,size)
    cx_point2 = np.random.randint(1,size-1)
    if cx_point2 >= cx_point1:
        cx_point2 += 1
    else:
        cx_point2, cx_point1 = cx_point1, cx_point2
    tmp1[cx_point1:cx_point2], tmp2[cx_point1:cx_point2] = tmp2[cx_point1:cx_point2], tmp1[cx_point1:cx_point2]
    new_child1 = Individual(tmp1)
    new_child2 = Individual(tmp2)
    return new_child1, new_child2

def crossover(selected):
    '''交叉の関数'''
    children = []
    if POPURATIONS % 2:
        selected.append(selected[0])
    for child1, child2 in zip(selected[::2], selected[1::2]):
        if np.random.rand() < CROSSOVER_PB:
            child1, child2 = cross_two_point_copy(child1, child2)
        children.append(child1)
        children.append(child2)
    children = children[:POPURATIONS]
    return children

def mutate(children):
    for child in children:
        if np.random.rand() < MUTATION_PB:
            child.mutate()
    return children


def ga_solve(generation):
    '''遺伝的アルゴリズムのソルバー
    return: 各世代の最高適応度をまとめたリスト、最低適応度をまとめたリスト'''
    best = []
    worst = []
    # --- Generation loop
    print('Generation loop start.')
    for i in range(GENERATIONS):
        # --- Step1. Print fitness in the generation
        best_ind = max(generation, key=Individual.get_fitness)
        best.append(best_ind.fitness) #maxを記録
        worst_ind = min(generation, key=Individual.get_fitness)
        worst.append(worst_ind.fitness)
        print("Generation: " + str(i) \
                + ": Best fitness: " + str(best_ind.fitness) \
                + ". Worst fitness: " + str(worst_ind.fitness))
        # --- Step2. Selection
        #selected = select_roulette(generation)
        selected = select_tournament(generation)
        # --- Step3. Crossover
        children = crossover(selected)
        # --- Step4. Mutation
        generation = mutate(children)

    print("Generation loop ended. The best individual: ")
    print(best_ind.genom)
    return best, worst

np.random.seed(seed=65)
# param
POPURATIONS = 100  # 1世代の個体数
GENOMS = 50             # 遺伝子の長さ(n)
GENERATIONS = 40  #世代数
CROSSOVER_PB = 0.8
MUTATION_PB = 0.1

# create first genetarion
generation = create_generation(POPURATIONS, GENOMS)

# solve
best, worst = ga_solve(generation)

#plot
fig, ax = plt.subplots()
ax.plot(best, label='max')
ax.plot(worst, label='min')
ax.axhline(y=GENOMS, color='black', linestyle=':', label='true')
ax.set_xlim([0, GENERATIONS - 1])
ax.set_ylim([0, GENOMS * 1.1])
ax.legend(loc='best')
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
ax.set_title('Tournament Select')
plt.show()