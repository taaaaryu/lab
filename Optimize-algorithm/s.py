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