from tian_neng.MergingAlgorithm import MergingAlgorithm
from tian_neng.VisualizeUtil import VisualizeUtil

if __name__ == '__main__':
    """
    说明：
        - 涉及字段：发运时间、发货地板块、收货地板块、收货地址、总重量（千克）
        - 规则：
            1. 运输价格计算：
                运输量>=25吨: 1C/(吨*米)
                25吨>运输量>=10吨: 1.01~1.1C/(吨*米)
                10吨>运输量: 1.1~1.2C/(吨*米)
            2. 每个订单最多延迟48h
            
        - 假设条件：
            1. 所有原始订单均为零担物流
    
    """

    import pandas as pd
    import numpy as np
    from AMapUtil import AMapUtil
    from tqdm import tqdm

    # 删除原有图表文件
    VisualizeUtil.delete_files('./graph')

    data_path = './data/0610_运单数据.xlsx'

    df = pd.read_excel(data_path)

    # 替换列名
    new_col_name_mapping = {
        "发运时间": "shipping_time",
        "完结时间": "completion_time",
        "运单号": "waybill_number",
        "电子运单号": "electronic_waybill_number",
        "销售组织分类": "sales_organization_category",
        "销售组织": "sales_module",
        "发货地板块": "shipping_region",
        "目的地板块": "destination_region",
        "销售组织.1": "sales_organization",
        "客户分类": "customer_category",
        "收货地址": "delivery_address",
        "总重量(千克)": "total_weight"
    }
    df = df.rename(columns=new_col_name_mapping)

    # TODO
    # 定义发货地
    shipping_region = "安徽省界首市田营工业区"

    # 替换值
    df["shipping_region"] = df["shipping_region"].replace("界首板块", shipping_region)

    # 选择列名
    chosen_cols = [
        "shipping_time",
        "shipping_region",
        "destination_region",
        "delivery_address",
        "total_weight"
    ]
    df = df.loc[:, chosen_cols]

    # 转换列格式
    df["shipping_time"] = pd.to_datetime(df["shipping_time"])

    # 按日期升序排序
    df = df.sort_values(by="shipping_time", ignore_index=True)

    # 将发货地和目的地相同的订单进行分组（发货地已相同）
    # 根据destination_region字段
    # 粒度为市级
    dfs = {group_name: data.reset_index(drop=True) for group_name, data in df.groupby("destination_region")}

    # 计算开车距离
    origin_coords = AMapUtil.get_geocode(shipping_region)  # 发货地点坐标
    driving_distance_mapping = dict(zip(dfs.keys(), [0] * len(dfs)))  # 目的地坐标映射
    for group_name in tqdm(driving_distance_mapping.keys(), desc='calculating driving distance...'):
        destination_coords = AMapUtil.get_geocode(group_name)
        distance = AMapUtil.get_driving_distance(origin_coords, destination_coords)
        driving_distance_mapping[group_name] = distance
        dfs[group_name]["driving_distance"] = [distance] * len(dfs[group_name])
    AMapUtil.save_local_temp()

    """
    状态变量：定义dp[j]表示处理到第j个订单时的最大合并发货量
    状态转移方程：
        dp[j]=max(订单j发货量+max{dp[i] | i<j and t_j - t_i <= 48h}, dp[j-1])
        第j个订单可以选择与之前某个i订单合并（需满足时间窗口），或单独发货
    时间矩阵：计算任意两个订单的时间差是否在48h内
    最优解推导：
        dp[0]=订单0发货量
        dp[1]=max(订单1发货量, 订单0+1发货量)
        dp[2]=max(订单2发货量, 订单1+2发货量, 订单0+1+2发货量)
        ...
    """

    # 对每个发货地和目的地相同的订单组从时间维度进行合并

    time_combined_df = pd.DataFrame(columns=[col for col in df.columns])
    sub_group_df_info_dict = {}  # 每个时间合并订单的信息（用于可视化）
    optimized_total_shipping_cost = 0
    for group_name, data in tqdm(dfs.items(), desc='merging orders...'):
        sub_group_df_list = MergingAlgorithm.dynamic_programming_merging(data=data)

        sub_group_df_info_list = []
        group_shipping_cost = 0
        for sub_group_df in sub_group_df_list:
            # 发货地和目的地相同，距离也相同，取第一个订单的距离
            valid_distance = sub_group_df.loc[0, "driving_distance"]
            order_weight = sub_group_df["total_weight"].sum()

            # 添加合并订单到新的df
            new_row = pd.DataFrame({
                "shipping_time": [sub_group_df.iloc[-1]["shipping_time"]],
                "shipping_region": [sub_group_df.iloc[-1]["shipping_region"]],
                "destination_region": [sub_group_df.iloc[-1]["destination_region"]],
                "delivery_address": [sub_group_df.iloc[-1]["delivery_address"]],
                "total_weight": [order_weight]
            })
            time_combined_df = pd.concat([time_combined_df, new_row], ignore_index=True)

            if order_weight >= 25 * 1000:
                unit_price = 1
            elif order_weight >= 10 * 1000:
                unit_price = 1.1
            else:
                unit_price = 1.2

            order_shipping_cost = unit_price * order_weight / 1000 * valid_distance
            group_shipping_cost += order_shipping_cost

            sub_group_df_info_list.append({
                "weight_sum": order_weight,
                "start_time": sub_group_df["shipping_time"].min(),
                "end_time": sub_group_df["shipping_time"].max(),
                "order_num": len(sub_group_df)
            })

        optimized_total_shipping_cost += group_shipping_cost
        sub_group_df_info_dict[group_name] = sub_group_df_info_list

    # 绘制合并订单甘特图
    VisualizeUtil.draw_total_gantt_chart(sub_group_df_info_dict)

    total_shipping_cost = 0
    for group_name, data in dfs.items():
        # 发货地和目的地相同，距离也相同，取第一个订单的距离
        valid_distance = data.loc[0, "driving_distance"]

        group_shipping_cost = 0
        for m in range(len(data)):
            order_weight = data.loc[m, "total_weight"]
    
            if order_weight >= 25 * 1000:
                unit_price = 1
            elif order_weight >= 10 * 1000:
                unit_price = 1.1
            else:
                unit_price = 1.2
    
            order_shipping_cost = unit_price * order_weight / 1000 * valid_distance
            group_shipping_cost += order_shipping_cost
        total_shipping_cost += group_shipping_cost

    optimized_cost_ratio = (total_shipping_cost - optimized_total_shipping_cost) / total_shipping_cost

    print(f"total shipping cost: {total_shipping_cost} C")
    print(f"optimized total shipping cost: {optimized_total_shipping_cost} C")
    print(f"optimized cost ratio: {optimized_cost_ratio * 100} %")

    pass





