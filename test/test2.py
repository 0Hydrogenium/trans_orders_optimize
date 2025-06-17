import math
import random
import numpy as np
from deap import base, creator, tools, algorithms
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm

# 模拟数据 - 假设我们有一些目的地（A,B,C,D）和起点O
locations = {
    'O': (0, 0),  # 起点/终点
    'A': (10, 0),
    'B': (0, 10),
    'C': (-10, 0),
    'D': (0, -10),
    'E': (15, 15),
    'F': (-15, -15),
    'G': (-5, 10),
    'H': (5, -10)
}

# 每个目的地的货物需求量（吨）
demands = {
    'A': 12,
    'B': 8,
    'C': 18,
    'D': 15,
    'E': 20,
    'F': 10,
    'G': 5,
    'H': 7
}

# 计算所有配送点的距离矩阵
distance_matrix = {}
for from_node in locations:
    distance_matrix[from_node] = {}
    for to_node in locations:
        dist = math.sqrt((locations[from_node][0] - locations[to_node][0]) ** 2 +
                         (locations[from_node][1] - locations[to_node][1]) ** 2)
        distance_matrix[from_node][to_node] = dist


def get_transport_rate(weight):
    """根据载重量确定运输费率"""
    if weight >= 25:
        return 1.0
    elif weight >= 10:
        return 1.1
    else:
        return 1.2


def calculate_route_cost(route):
    """计算单条路线的成本"""
    if not route:  # 空路线
        return 0

    total_cost = 0
    current_load = sum(demands[node] for node in route)
    segments = []

    # 创建路线段 (O -> 第一站, 各站之间, 最后一站 -> O)
    segments.append(('O', route[0]))
    for i in range(len(route) - 1):
        segments.append((route[i], route[i + 1]))
    segments.append((route[-1], 'O'))

    # 计算每段成本
    for start, end in segments:
        dist = distance_matrix[start][end]
        rental_cost = 0.2 * dist  # 租赁费用

        # 运输成本（只有当有载重时才计算）
        if current_load > 0:
            rate = get_transport_rate(current_load)
            transport_cost = current_load * dist * rate
        else:
            transport_cost = 0

        total_cost += rental_cost + transport_cost

        # 到达目的地后卸货（除了返回O点）
        if end != 'O' and end in demands:
            current_load -= demands.get(end, 0)

    return total_cost


def evaluate(individual):
    """评估个体的适应度（成本越低越好）"""
    # 解码个体为路线集合
    routes = decode_individual(individual)

    total_cost = 0
    covered = set()

    # 计算总成本
    for route in routes:
        # 确保路线有效且非空
        if not route:
            continue

        # 计算路线成本
        route_cost = calculate_route_cost(route)
        total_cost += route_cost

        # 记录覆盖的目的地
        covered.update(route)

    # 添加未覆盖目的地的惩罚（每个未覆盖地点增加10000元成本）
    penalty = 10000 * (len(demands) - len(covered))
    total_cost += penalty

    return total_cost,


def decode_individual(individual):
    """将个体（目的地列表）解码为路线集合"""
    routes = []
    current_route = []

    for node in individual:
        if node == 'O' and current_route:
            routes.append(current_route)
            current_route = []
        elif node != 'O':
            current_route.append(node)

    # 添加最后一条路线（如果有）
    if current_route:
        routes.append(current_route)

    return routes


def create_individual():
    """创建个体：所有目的地的随机排列，包含分隔符"""
    all_nodes = list(demands.keys())

    # 决定使用多少辆车（1到目的地点数量的1/2）
    num_vehicles = random.randint(1, max(1, len(all_nodes) // 2))

    # 添加车辆分隔符
    separators = ['O'] * (num_vehicles)

    individual = all_nodes + separators
    random.shuffle(individual)
    return individual


def cxPartialyMatched(ind1, ind2):
    """部分匹配交叉（PMX）"""
    # 确保两个个体长度相同
    size = min(len(ind1), len(ind2))
    cxpoint1 = random.randint(0, size - 1)
    cxpoint2 = random.randint(cxpoint1 + 1, size)

    # 映射段
    segment1 = ind1[cxpoint1:cxpoint2]
    segment2 = ind2[cxpoint1:cxpoint2]

    # 创建映射
    mapping1 = {}
    mapping2 = {}
    for a, b in zip(segment1, segment2):
        mapping1[a] = b
        mapping2[b] = a

    # 交换中间段
    ind1[cxpoint1:cxpoint2] = segment2
    ind2[cxpoint1:cxpoint2] = segment1

    # 修复冲突
    for i in range(len(ind1)):
        if i < cxpoint1 or i >= cxpoint2:
            # 解决冲突
            while ind1[i] in segment2:
                ind1[i] = mapping1.get(ind1[i], ind1[i])
            while ind2[i] in segment1:
                ind2[i] = mapping2.get(ind2[i], ind2[i])

    return ind1, ind2


def mutate_individual(individual, indpb):
    """个体变异操作"""
    # 交换变异
    for i in range(len(individual)):
        if random.random() < indpb:
            j = random.randint(0, len(individual) - 1)
            individual[i], individual[j] = individual[j], individual[i]

    # 分隔符变异
    if random.random() < indpb:
        # 添加或删除分隔符
        if random.random() < 0.5 and individual.count('O') < len(demands) - 1:
            # 添加分隔符
            positions = [idx for idx, gene in enumerate(individual) if gene != 'O']
            if positions:
                pos = random.choice(positions)
                individual.insert(pos, 'O')
        elif 'O' in individual:
            # 移除分隔符
            positions = [idx for idx, gene in enumerate(individual) if gene == 'O']
            if positions:
                pos = random.choice(positions)
                del individual[pos]

    # 删除个体的适应度值，因为已经改变了
    del individual.fitness.values

    return individual,


def print_solution(best_ind):
    """打印最佳解决方案"""
    best_routes = decode_individual(best_ind)
    total_cost = 0
    route_demands = []
    covered = set()

    print("\n======= 最佳解决方案 =======")
    for i, route in enumerate(best_routes):
        # 确保路线有效且非空
        if not route:
            continue

        # 计算路线成本
        route_cost = calculate_route_cost(route)
        route_demand = sum(demands[node] for node in route)
        total_cost += route_cost
        route_demands.append(route_demand)
        covered.update(route)

        # 打印路线
        print(f"路线 {i + 1}: {' -> '.join(['O'] + route + ['O'])}")
        print(f"  需求: {route_demand} 吨 | 成本: {route_cost:.2f}元")

    # 检查覆盖情况
    print("\n===== 总结 =====")
    print(f"总成本: {total_cost:.2f}元")
    print(f"使用车辆数: {len(best_routes)}")
    print(f"需求覆盖率: {len(covered)}/{len(demands)} 目的地")

    if covered == set(demands.keys()):
        print("成功覆盖所有目的地!")
    else:
        missing = set(demands.keys()) - covered
        print(f"警告: 缺少目的地 - {missing}")


# 设置遗传算法
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", cxPartialyMatched)
toolbox.register("mutate", mutate_individual, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


# 运行遗传算法
def main():
    random.seed(42)
    pop = toolbox.population(n=100)

    # 评估初始种群
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # 创建Hall of Fame
    hof = tools.HallOfFame(1)
    hof.update(pop)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # 运行算法 - 手动实现类似eaSimple的算法
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields

    for gen in tqdm(range(50), desc="evolving..."):  # 进化50代
        # 选择下一代个体
        offspring = toolbox.select(pop, len(pop))
        # 克隆选择出的个体
        offspring = list(map(toolbox.clone, offspring))

        # 交叉操作
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # 变异操作
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # 评估个体（仅评估那些改变了或者没有适应度的）
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 更新种群
        pop[:] = offspring

        # 更新Hall of Fame和统计信息
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        print(logbook.stream)

    return pop, logbook, hof


if __name__ == "__main__":
    pop, log, hof = main()
    best_ind = hof[0]

    # 打印最佳解决方案
    print_solution(best_ind)

    # 绘制进化过程
    gen = log.select("gen")
    min_fit = log.select("min")
    avg_fit = log.select("avg")

    plt.figure(figsize=(10, 6))
    plt.plot(gen, min_fit, 'b-', label="最低成本")
    plt.plot(gen, avg_fit, 'r-', label="平均成本")
    plt.xlabel("代数")
    plt.ylabel("成本")
    plt.title("成本随代数的变化")
    plt.legend()
    plt.grid(True)
    plt.show()