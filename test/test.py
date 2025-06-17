if __name__ == '__main__':
    import math
    import random
    from copy import deepcopy

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
        """计算单条路线的成本（修正载重计算）"""
        if not route:  # 空路线
            return 0

        total_cost = 0
        current_load = 0
        segments = []

        # 创建路线段 (O -> 第一站, 各站之间, 最后一站 -> O)
        segments.append(('O', route[0]))
        for i in range(len(route) - 1):
            segments.append((route[i], route[i + 1]))
        segments.append((route[-1], 'O'))

        # 计算起始装载量（路线所有需求之和）
        route_demand = sum(demands[node] for node in route)

        # 从起点出发时装载所有货物
        current_load = route_demand

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
            if end != 'O':
                current_load -= demands.get(end, 0)

        return total_cost


    class Chromosome:
        """表示一个潜在解决方案的染色体"""

        def __init__(self, genes=None):
            # 基因结构: [[路线1], [路线2], ...] 每条路线包含目的地的有序列表
            self.genes = genes if genes is not None else []
            self.fitness = 0
            self.total_cost = float('inf')

        def calculate_fitness(self):
            """计算整个解决方案的成本并转换为适应度（修正）"""
            # 检查是否覆盖所有目的地
            covered = set()
            for route in self.genes:
                for node in route:
                    covered.add(node)

            # 计算总成本
            self.total_cost = 0
            for route in self.genes:
                self.total_cost += calculate_route_cost(route)

            # 添加未覆盖目的地的惩罚（每个未覆盖地点增加10000元成本）
            penalty = 10000 * (len(demands) - len(covered))
            self.total_cost += penalty

            # 适应度为总成本的倒数（成本越低，适应度越高）
            self.fitness = 1 / (self.total_cost + 1e-6)  # 避免除以0

        def __str__(self):
            return f"Routes: {self.genes}, Cost: {self.total_cost:.2f}"


    class GeneticAlgorithm:
        """遗传算法实现（修正版本）"""

        def __init__(self, destinations, population_size=100, elite_size=10,
                     mutation_rate=0.15, max_generations=200):
            self.destinations = destinations  # 所有需要访问的目的地
            self.population_size = population_size
            self.elite_size = elite_size
            self.mutation_rate = mutation_rate
            self.max_generations = max_generations
            self.best_solution = None

        def initial_population(self):
            """创建初始种群（增强多样性）"""
            population = []

            # 1. 每个目的地单独成路线
            for _ in range(self.population_size // 4):
                individual = [[dest] for dest in self.destinations]
                population.append(Chromosome(individual))

            # 2. 将所有目的地随机划分为1-4条路线
            for _ in range(self.population_size // 4):
                shuffled = random.sample(self.destinations, len(self.destinations))
                num_routes = random.randint(1, min(4, len(self.destinations)))

                # 划分目的地到不同路线
                routes = []
                idx = 0
                step = len(shuffled) // num_routes
                for i in range(num_routes):
                    start_idx = i * step
                    end_idx = (i + 1) * step if i < num_routes - 1 else len(shuffled)
                    if start_idx < end_idx:
                        routes.append(shuffled[start_idx:end_idx])
                population.append(Chromosome(routes))

            # 3. 使用最近邻算法生成路线
            for _ in range(self.population_size // 4):
                unvisited = set(self.destinations)
                routes = []

                while unvisited:
                    current_route = []
                    current_load = 0
                    current_node = 'O'

                    # 尝试填充当前路线
                    while unvisited and current_load <= 30:  # 假设车辆最大载重30吨
                        # 寻找最近的未访问节点
                        best_dist = float('inf')
                        best_node = None

                        for node in unvisited:
                            if current_load + demands[node] > 30:  # 超过载重限制
                                continue
                            dist = distance_matrix[current_node][node]
                            if dist < best_dist:
                                best_dist = dist
                                best_node = node

                        if best_node:
                            current_route.append(best_node)
                            unvisited.remove(best_node)
                            current_load += demands[best_node]
                            current_node = best_node
                        else:
                            break  # 没有合适的节点

                    if current_route:
                        routes.append(current_route)

                population.append(Chromosome(routes))

            # 4. 完全随机划分
            for _ in range(self.population_size - len(population)):
                shuffled = random.sample(self.destinations, len(self.destinations))
                num_routes = random.randint(1, len(self.destinations))
                routes = []
                idx = 0
                for _ in range(num_routes):
                    if idx >= len(shuffled):
                        break
                    route_len = random.randint(1, len(shuffled) - idx)
                    routes.append(shuffled[idx:idx + route_len])
                    idx += route_len
                population.append(Chromosome(routes))

            return population

        def rank_routes(self, population):
            """根据适应度对种群排序"""
            for chromosome in population:
                if chromosome.fitness == 0:
                    chromosome.calculate_fitness()

            population.sort(key=lambda x: x.fitness, reverse=True)
            return population

        def selection(self, ranked_population):
            """选择操作：锦标赛选择与精英保留结合"""
            selection_results = []

            # 保留精英个体
            selection_results.extend(ranked_population[:self.elite_size])

            # 锦标赛选择
            tournament_size = 5
            for _ in range(self.population_size - self.elite_size):
                tournament = random.sample(ranked_population, tournament_size)
                winner = max(tournament, key=lambda x: x.fitness)
                selection_results.append(deepcopy(winner))

            return selection_results

        def crossover(self, parent1, parent2):
            """交叉操作：顺序交叉"""
            child_routes = []

            # 随机选择从每个父代复制的路线数量
            num_routes1 = random.randint(1, min(3, len(parent1.genes)))
            selected1 = random.sample(parent1.genes, num_routes1)

            # 从父代1复制选定的路线
            for route in selected1:
                child_routes.append(route)

            # 收集已覆盖的目的地
            covered = set()
            for route in child_routes:
                covered.update(route)

            # 从父代2添加未被覆盖的目的地
            remaining_dests = set(self.destinations) - covered

            if remaining_dests:
                # 从父代2选择包含未覆盖目的地的路线
                for route in parent2.genes:
                    # 只添加包含未覆盖目的地的节点
                    new_route = [node for node in route if node in remaining_dests]
                    if new_route:
                        child_routes.append(new_route)
                        covered.update(new_route)
                        remaining_dests -= set(new_route)

            # 如果还有剩余目的地，创建新的路线
            if remaining_dests:
                while remaining_dests:
                    # 尝试创建包含多个点的路线
                    route_len = min(len(remaining_dests), random.randint(1, 3))
                    route = random.sample(list(remaining_dests), route_len)
                    child_routes.append(route)
                    remaining_dests -= set(route)

            return Chromosome(child_routes)

        def mutate(self, chromosome):
            """变异操作 - 更保守的实现"""
            genes = deepcopy(chromosome.genes)

            # 仅当需要变异时执行
            if random.random() < self.mutation_rate:
                mutation_type = random.choice([
                    'swap', 'move', 'split', 'merge', 'reverse', 'reorder'
                ])

                if mutation_type == 'swap' and len(genes) > 1:
                    # 交换两条路线
                    idx1, idx2 = random.sample(range(len(genes)), 2)
                    genes[idx1], genes[idx2] = genes[idx2], genes[idx1]

                elif mutation_type == 'move' and len(genes) > 1:
                    # 在路线间移动节点
                    src_idx = random.randint(0, len(genes) - 1)
                    if not genes[src_idx]:
                        return

                    dest_idx = random.randint(0, len(genes) - 1)
                    while dest_idx == src_idx and len(genes) > 1:
                        dest_idx = random.randint(0, len(genes) - 1)

                    node_idx = random.randint(0, len(genes[src_idx]) - 1)
                    node = genes[src_idx].pop(node_idx)

                    if not genes[src_idx]:
                        del genes[src_idx]
                        if dest_idx > src_idx:
                            dest_idx -= 1

                    if not genes:  # 如果所有路线被删除，创建新路线
                        genes.append([node])
                    else:
                        if dest_idx >= len(genes):  # 确保目标索引有效
                            dest_idx = len(genes) - 1
                        insert_idx = random.randint(0, len(genes[dest_idx]))
                        genes[dest_idx].insert(insert_idx, node)

                elif mutation_type == 'split' and genes:
                    # 拆分路线
                    route_idx = random.randint(0, len(genes) - 1)
                    if len(genes[route_idx]) > 1:
                        split_point = random.randint(1, len(genes[route_idx]) - 1)
                        new_route = genes[route_idx][split_point:]
                        genes[route_idx] = genes[route_idx][:split_point]
                        genes.append(new_route)

                elif mutation_type == 'merge' and len(genes) > 1:
                    # 合并路线
                    idx1, idx2 = random.sample(range(len(genes)), 2)
                    genes[idx1].extend(genes[idx2])
                    del genes[idx2]

                elif mutation_type == 'reverse' and genes:
                    # 反转路线顺序
                    route_idx = random.randint(0, len(genes) - 1)
                    if len(genes[route_idx]) > 1:
                        genes[route_idx] = list(reversed(genes[route_idx]))

                elif mutation_type == 'reorder' and genes:
                    # 重新排序路线内节点
                    route_idx = random.randint(0, len(genes) - 1)
                    if len(genes[route_idx]) > 1:
                        random.shuffle(genes[route_idx])

            chromosome.genes = genes

        def evolve(self):
            """执行遗传算法进化（带跟踪）"""
            population = self.initial_population()
            print(f"Initial population created with {len(population)} individuals")

            best_cost = float('inf')

            for generation in range(self.max_generations):
                # 评估种群
                ranked_population = self.rank_routes(population)
                current_best = ranked_population[0]

                # 更新全局最优
                if current_best.total_cost < best_cost:
                    self.best_solution = deepcopy(current_best)
                    best_cost = current_best.total_cost
                    print(f"Generation {generation + 1}: New best solution with cost {best_cost:.2f}")

                # 显示进展
                avg_fitness = sum(c.fitness for c in population) / len(population)
                print(f"Gen {generation + 1}: Best Cost = {best_cost:.2f}, Avg Fitness = {avg_fitness:.6f}")

                # 选择
                selected = self.selection(ranked_population)

                # 交叉和变异
                children = selected[:self.elite_size]  # 保留精英

                for _ in range(self.population_size - self.elite_size):
                    parent1, parent2 = random.sample(selected, 2)

                    # 交叉
                    child = self.crossover(parent1, parent2)

                    # 变异
                    self.mutate(child)

                    children.append(child)

                population = children

            return self.best_solution


    # 执行遗传算法
    destinations = list(demands.keys())  # 所有需要访问的目的地（不包括O点）
    ga = GeneticAlgorithm(
        destinations=destinations,
        population_size=100,
        elite_size=15,
        mutation_rate=0.15,
        max_generations=150
    )

    best_solution = ga.evolve()

    # 输出最终结果
    print("\nOptimal Solution Found:")
    print(best_solution)

    # 打印详细路线成本
    total_cost = 0
    for i, route in enumerate(best_solution.genes):
        cost = calculate_route_cost(route)
        total_cost += cost
        route_demand = sum(demands[loc] for loc in route)
        print(f"Route {i + 1}: {route} | Demand: {route_demand} tons | Cost: {cost:.2f}")

    print(f"\nTotal Cost: {total_cost:.2f}")
    print(f"Total Vehicles Used: {len(best_solution.genes)}")

    # 检查是否覆盖所有目的地
    covered = set(dest for route in best_solution.genes for dest in route)
    if covered == set(destinations):
        print("All destinations covered!")
    else:
        missing = set(destinations) - covered
        print(f"Warning: Missing destinations - {missing}")
        print(f"Coverage: {len(covered)}/{len(destinations)}")