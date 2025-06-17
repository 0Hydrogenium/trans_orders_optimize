import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops, degree
from collections import deque
import random


# 环境设置
class VehicleRoutingEnv:
    def __init__(self, points, demands, o_point=(0, 0)):
        self.o_point = np.array(o_point)
        self.points = np.array(points)  # 目的地点坐标
        self.demands = np.array(demands)  # 各目的地需求
        self.n_points = len(points)
        self.reset()

    def reset(self):
        self.visited = np.zeros(self.n_points, dtype=bool)
        self.current_position = self.o_point
        self.current_vehicle_path = []
        self.current_load = 0.0
        self.total_cost = 0.0
        self.done = False
        self.vehicle_count = 0
        return self._get_state()

    def _get_state(self):
        """状态表示: [位置, 需求, 访问状态]"""
        state = []
        # 添加O点
        state.append([self.o_point[0], self.o_point[1], 0.0,
                      1 if np.array_equal(self.current_position, self.o_point) else 0])
        # 添加目的地点
        for i in range(self.n_points):
            point = self.points[i]
            state.append([point[0], point[1], self.demands[i],
                          1 if not self.visited[i] else 0])  # 1表示未访问
        return np.array(state, dtype=np.float32)

    def _distance(self, a, b):
        return np.linalg.norm(a - b)

    def _calculate_leg_cost(self, start, end, load):
        dist = self._distance(start, end)
        if load >= 25:
            rate = 1.0
        elif load >= 10:
            rate = 1.1
        else:
            rate = 1.2
        transport_cost = load * dist * rate
        rent_cost = dist * 0.2
        return transport_cost + rent_cost

    def _finish_vehicle_route(self):
        if not self.current_vehicle_path:
            return 0.0, []

        # 车辆完整路径: O -> [路径点] -> O
        points_in_route = self.points[self.current_vehicle_path]
        demands_in_route = self.demands[self.current_vehicle_path]
        total_demand = sum(demands_in_route)

        legs = []
        legs.append((self.o_point, points_in_route[0]))
        for i in range(len(points_in_route) - 1):
            legs.append((points_in_route[i], points_in_route[i + 1]))
        legs.append((points_in_route[-1], self.o_point))

        cost = 0.0
        current_load = total_demand
        # 计算各路段成本
        for i, (start, end) in enumerate(legs):
            cost += self._calculate_leg_cost(start, end, current_load)
            if i < len(points_in_route):
                current_load -= demands_in_route[i]
        return cost, legs

    def step(self, action):
        # 处理结束动作（返回O点）
        if action == self.n_points:  # 结束动作索引 = 目的地数量
            vehicle_cost, legs = self._finish_vehicle_route()
            self.total_cost += vehicle_cost
            self.current_vehicle_path = []
            self.current_load = 0.0
            self.current_position = self.o_point
            self.vehicle_count += 1

            # 检查是否所有点都访问过
            if np.all(self.visited):
                self.done = True
                return self._get_state(), -self.total_cost, True, {}
            return self._get_state(), -vehicle_cost, False, {'legs': legs}

        # 处理选择目的地的动作
        if self.visited[action]:
            raise ValueError("Destination already visited")

        self.visited[action] = True
        next_point = self.points[action]
        self.current_vehicle_path.append(action)

        # 计算到新目的地的成本
        start_pos = self.current_position
        end_pos = next_point

        if self.current_load == 0:  # 从O点出发或第一次装货
            # 车辆装载所有后续需求
            self.current_load = sum(self.demands[self.current_vehicle_path])

        cost = self._calculate_leg_cost(start_pos, end_pos, self.current_load)
        self.total_cost += cost
        self.current_load -= self.demands[action]
        self.current_position = next_point

        return self._get_state(), -cost, False, {'leg': (start_pos, end_pos)}

    def get_valid_actions(self):
        """获取当前有效动作列表"""
        valid_actions = []

        # 添加未访问目的地的动作
        for i in range(self.n_points):
            if not self.visited[i]:
                valid_actions.append(i)

        # 添加结束动作（如果有路径需要结束）
        if len(self.current_vehicle_path) > 0:
            valid_actions.append(self.n_points)  # 结束动作索引

        return valid_actions


# 图神经网络模块
class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # 添加自环
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # 线性变换
        x = self.lin(x)
        # 归一化
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # 传播
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


# 策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 节点特征转换
        self.node_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 图卷积层
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # 聚合层
        self.global_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 输出层
        self.action_out = nn.Linear(hidden_dim * 2, 1)
        self.value_out = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 节点嵌入
        node_embeds = self.node_embed(x)

        # GCN传播
        node_embeds = F.relu(self.conv1(node_embeds, edge_index))
        node_embeds = F.relu(self.conv2(node_embeds, edge_index))

        # 全局上下文 (平均池化)
        global_ctx = torch.mean(node_embeds, dim=0)
        global_ctx = self.global_embed(global_ctx).unsqueeze(0)

        # 计算每个节点作为动作的分数
        action_scores = []
        for i in range(node_embeds.size(0)):
            node_embed = node_embeds[i]
            combined = torch.cat([node_embed, global_ctx.squeeze(0)], dim=-1)
            action_scores.append(self.action_out(combined))

        # 结束动作分数（特殊动作）
        end_action_score = self.action_out(torch.cat([global_ctx.squeeze(0), global_ctx.squeeze(0)], dim=-1))
        action_scores.append(end_action_score)
        action_scores = torch.cat(action_scores)

        # 状态值函数估计
        state_value = self.value_out(global_ctx)

        return action_scores, state_value


# 完整的REINFORCE代理
class ReinforceAgent:
    def __init__(self, input_dim, hidden_dim, lr=1e-4, gamma=0.99):
        self.policy_net = PolicyNetwork(input_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.saved_log_probs = []
        self.rewards = []
        self.state_values = []
        self.entropies = []
        self.history_returns = deque(maxlen=100)
        self.baseline = 0.0

    def select_action(self, state, env, train=True):
        # 获取有效动作
        valid_actions = env.get_valid_actions()

        # 如果没有有效动作，返回结束动作
        if not valid_actions:
            return env.n_points

        # 构建图数据结构
        node_features = state
        n_nodes = node_features.shape[0]

        # 完全连接图的边
        edge_index = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=edge_index
        )

        # 获取动作分数
        with torch.set_grad_enabled(train):
            action_scores, state_value = self.policy_net(data)

            # 创建掩码屏蔽无效动作
            mask = torch.zeros_like(action_scores, dtype=torch.bool)
            # 只允许有效动作
            for idx in valid_actions:
                # 目的地动作
                if idx < env.n_points:
                    mask[idx + 1] = 1  # +1 因为状态索引0是O点
                # 结束动作
                else:
                    mask[-1] = 1

            # 如果没有有效动作，则强制执行结束动作
            if torch.sum(mask) == 0:
                mask[-1] = 1

            action_scores[~mask] = -float('inf')
            probs = F.softmax(action_scores, dim=-1)
            dist = torch.distributions.Categorical(probs)

            if train:
                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()
                self.saved_log_probs.append(log_prob)
                self.state_values.append(state_value)
                self.entropies.append(entropy)
            else:
                action = torch.argmax(probs)

            # 将动作索引映射回环境动作
            if action.item() == action_scores.size(0) - 1:  # 结束动作
                return env.n_points
            else:
                return action.item() - 1  # -1 因为状态索引0是O点

    def finish_episode(self):
        if not self.rewards:
            return 0.0

        R = 0
        policy_loss = []
        value_loss = []
        returns = []

        # 计算折扣回报
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32)
        # 更新基线
        self.history_returns.extend(returns.tolist())
        self.baseline = np.mean(self.history_returns) if self.history_returns else 0

        # 标准化回报
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        else:
            returns = returns - returns.mean()

        # 计算损失
        for log_prob, value, R in zip(self.saved_log_probs, self.state_values, returns):
            advantage = R - self.baseline
            policy_loss.append(-log_prob * advantage.detach())
            value_loss.append(F.mse_loss(value.squeeze(), torch.tensor([R])))

        # 添加熵正则化
        entropy_loss = -0.01 * sum(self.entropies) if self.entropies else torch.tensor(0.0)

        total_loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum() + entropy_loss

        # 更新网络
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # 清除缓存
        self.saved_log_probs = []
        self.rewards = []
        self.state_values = []
        self.entropies = []

        return total_loss.item()


# 训练函数
def train(num_episodes, env, agent):
    for i_episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        step_count = 0
        while not env.done:
            action = agent.select_action(state, env)
            next_state, reward, done, _ = env.step(action)
            agent.rewards.append(reward)
            state = next_state
            episode_reward += reward
            step_count += 1

            if done:
                agent.rewards.append(reward)
                break

        loss = agent.finish_episode()
        if i_episode % 10 == 0:
            print(
                f'Episode {i_episode}, Loss: {loss:.2f}, Reward: {episode_reward:.2f}, Steps: {step_count}, Vehicles: {env.vehicle_count}')


# 测试函数
def test(env, agent):
    state = env.reset()
    total_reward = 0
    legs = []
    step_count = 0
    while not env.done:
        action = agent.select_action(state, env, train=False)
        next_state, reward, done, info = env.step(action)
        state = next_state
        total_reward += reward
        step_count += 1

        if 'legs' in info:
            legs.append(info['legs'])
        elif 'leg' in info:
            legs.append([info['leg']])

    print(f"Total cost: {-total_reward:.2f}, Steps: {step_count}, Vehicles used: {env.vehicle_count}")
    for i, vehicle_legs in enumerate(legs):
        print(f"Vehicle {i + 1} route:")
        for leg in vehicle_legs:
            start, end = leg
            print(f"  From ({start[0]:.1f}, {start[1]:.1f}) to ({end[0]:.1f}, {end[1]:.1f})")
    return total_reward


# 主程序
if __name__ == "__main__":
    # 固定随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 示例数据 - 5个目的地
    o_point = (0, 0)
    points = [(10, 20), (20, 30), (30, 40), (40, 50), (15, 25)]  # 目的地经纬度
    demands = [12, 8, 15, 20, 10]  # 各目的地货物量（吨）

    # 创建环境
    env = VehicleRoutingEnv(points, demands, o_point)
    input_dim = 4  # 状态特征维度: [经度, 纬度, 需求, 状态标志]
    hidden_dim = 128

    # 创建代理
    agent = ReinforceAgent(input_dim, hidden_dim, lr=1e-3)

    # 训练
    print("开始训练...")
    train(num_episodes=100, env=env, agent=agent)

    # 测试
    print("\n开始测试...")
    test_env = VehicleRoutingEnv(points, demands, o_point)
    total_reward = test(test_env, agent)