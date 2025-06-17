import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns
import os
import shutil

# 用于显示中文和负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class VisualizeUtil:

    @classmethod
    def delete_files(cls, folder_path):
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path)  # 递归删除整个文件夹
            except Exception as e:
                print(f"删除失败: {e}")

        # 重新创建空文件夹
        os.makedirs(folder_path, exist_ok=True)

    @classmethod
    def draw_total_gantt_chart(cls, info_dict):
        """
        用甘特图可视化合并后的订单时间分布
        """

        base_path = './graph/after_combined_gantt_chart'
        os.makedirs(base_path, exist_ok=True)

        total_info_list = []
        for group_name, info_list in info_dict.items():
            for info in info_list:
                info["destination"] = group_name
                total_info_list.append(info)

        max_line = 10
        last = 1 if len(total_info_list) % max_line > 0 else 0
        group_total_info_list = [total_info_list[i*max_line: (i+1)*max_line] for i in range(len(total_info_list) // max_line + last)]

        print(f"total {len(group_total_info_list)} gantt charts")
        for group_idx, info_list in enumerate(group_total_info_list):

            plt.clf()

            # 最多绘制20张代表性的图
            if group_idx >= 20:
                break

            plt.figure(figsize=(18, 14))
            # 设置颜色方案
            colors = [
                "#EBB2B3",
                "#B9DAE5",
                "#E1BCA7",
                "#F3DE95"
            ]

            max_weight_sum = max([info["weight_sum"] for info in info_list])
            pos = 0
            for info_idx, info in enumerate(info_list):
                bar_height = 0.3 + (info["weight_sum"] / max_weight_sum) * 0.4  # 条高在0.3-0.7之间变化

                # 绘制甘特图条
                plt.barh(
                    pos,
                    (info["end_time"] - info["start_time"]).days,  # 宽度为天数
                    left=info["start_time"],
                    height=bar_height,
                    color=colors[info_idx % len(colors)],
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=2
                )

                # 在条上标注订单数量和总重量
                # 计算条的中间x位置
                bar_center_x = info["start_time"] + timedelta(
                    days=(info["end_time"] - info["start_time"]).days) / 2
                # 计算条的顶部y位置并加上一些间距
                bar_top_y = pos + bar_height / 2 + 0.1

                plt.text(
                    bar_center_x,  # x位置：条的中间
                    bar_top_y,  # y位置：条的顶部上方
                    f"{info['order_num']}单\n{info['weight_sum']:.2f}kg",
                    ha='center',  # 水平居中对齐
                    va='bottom',  # 垂直底部对齐
                    fontsize=12,
                    color='black'
                )

                pos += 1

            # 设置坐标轴
            plt.yticks(
                [i for i in range(pos)],
                [info["destination"] for info in info_list],
                fontsize=12
            )

            plt.xticks(rotation=45)

            # 获取数据中的最早和最晚日期
            if info_list:
                all_dates = [date for info in info_list for date in [info["start_time"], info["end_time"]]]
                min_date = min(all_dates)
                max_date = max(all_dates)

                # 设置x轴日期范围
                plt.xlim(min_date - timedelta(days=1), max_date + timedelta(days=1))

                # 设置x轴为主刻度（每隔2天）
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

                # 设置x轴为次刻度（每天），用于显示网格
                plt.gca().xaxis.set_minor_locator(mdates.DayLocator(interval=1))

                # 添加网格线
                plt.grid(True, axis='x', linestyle='--', alpha=0.7)

            # 添加标签
            plt.xlabel('发货时间', fontsize=16, labelpad=30)
            plt.ylabel('目的地', fontsize=16, labelpad=30)

            plt.tight_layout()

            plt.savefig(f"{base_path}/gantt_chart_{group_idx}.png", dpi=300)
