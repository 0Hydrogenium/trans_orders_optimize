import numpy as np
import pandas as pd


class MergingAlgorithm:
    @classmethod
    def dynamic_programming_merging(cls, data):
        # 按发货时间升序排序
        sorted_data = data.sort_values("shipping_time").reset_index(drop=True)
        n = len(sorted_data)

        # 动态规划初始化
        dp = [0.0] * n  # dp[j]: 以j结尾的合并组最大总重量
        start = [0] * n  # start[j]: j所在合并组的最早订单索引
        prev = [-1] * n  # prev[j]: j在组内的前一个订单索引(-1表示无前驱)

        # 填充动态规划表
        for j in range(n):
            # 初始状态：j独立成组
            dp[j] = sorted_data.loc[j, "total_weight"]
            start[j] = j
            prev[j] = -1

            # 尝试将j合并到前面的组(i)
            for i in range(j):
                # 计算当前组的时间跨度（最早订单到j的时间差）
                time_diff = (sorted_data.loc[j, "shipping_time"] - sorted_data.loc[start[i], "shipping_time"]).days

                if time_diff <= 2:  # 关键修复：检查整个组的时间跨度
                    merged_weight = dp[i] + sorted_data.loc[j, "total_weight"]

                    # 如果合并后总重量更大，则更新状态
                    if merged_weight > dp[j]:
                        dp[j] = merged_weight
                        prev[j] = i
                        start[j] = start[i]  # 继承组的最早订单

        # 回溯构建分组
        groups = []
        processed = set()

        for j in range(n - 1, -1, -1):  # 从后向前遍历
            if j not in processed:
                current = j
                group_indices = []

                # 回溯组内所有订单
                while current != -1:
                    group_indices.append(current)
                    processed.add(current)
                    current = prev[current]  # 移至前一个订单

                # 按原始时间排序组内订单
                group_indices.sort()
                groups.append(sorted_data.iloc[group_indices].reset_index(drop=True))

        # 反转分组顺序（时间早的组在前）
        groups.reverse()

        return groups
