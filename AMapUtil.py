import os.path

import requests
import json
import pandas as pd
import numpy as np


class AMapUtil:

    TEMP_GEOCODE_SAVE_PATH = "./temp_geocode.csv"
    TEMP_DRIVING_DISTANCE_SAVE_PATH = "./temp_driving_distance.csv"

    API_DRIVING_URL = "https://restapi.amap.com/v3/direction/driving"  # 高德地图路径规划API
    API_GEO_URL = "https://restapi.amap.com/v3/geocode/geo"  # 高德地图地理编码API
    API_KEY = "ca72043adab007e07d1af7736eb2de17"

    is_init = False
    geocode_df = None
    driving_distance_df = None
    # 使用缓存字典加速查找
    geocode_cache = {}
    driving_distance_cache = {}

    @classmethod
    def init(cls):
        if not os.path.exists(cls.TEMP_GEOCODE_SAVE_PATH):
            pd.DataFrame(columns=["address", "lng,lat"]).to_csv(cls.TEMP_GEOCODE_SAVE_PATH, index=False)
        if not os.path.exists(cls.TEMP_DRIVING_DISTANCE_SAVE_PATH):
            pd.DataFrame(columns=["origin", "destination", "distance"]).to_csv(cls.TEMP_DRIVING_DISTANCE_SAVE_PATH, index=False)

        cls.geocode_df = pd.read_csv(cls.TEMP_GEOCODE_SAVE_PATH)
        cls.driving_distance_df = pd.read_csv(cls.TEMP_DRIVING_DISTANCE_SAVE_PATH)

        # 初始化缓存字典
        cls._init_cache()

        cls.is_init = True

    @classmethod
    def _init_cache(cls):
        # 从DataFrame构建缓存字典
        if cls.geocode_df is not None and not cls.geocode_df.empty:
            cls.geocode_cache = dict(zip(cls.geocode_df['address'], cls.geocode_df['lng,lat']))

        if cls.driving_distance_df is not None and not cls.driving_distance_df.empty:
            # 使用元组作为键 (origin, destination)
            cls.driving_distance_cache = {(row['origin'], row['destination']): row['distance']
                                          for _, row in cls.driving_distance_df.iterrows()}

    @classmethod
    def save_local_temp(cls):
        cls.geocode_df.to_csv(cls.TEMP_GEOCODE_SAVE_PATH, index=False)
        cls.driving_distance_df.to_csv(cls.TEMP_DRIVING_DISTANCE_SAVE_PATH, index=False)

    @classmethod
    def get_driving_distance(cls, origin, destination):
        """
        计算高德地图两个地点之间的开车距离

        :param origin: 起点坐标，格式为"经度,纬度"
        :param destination: 终点坐标，格式为"经度,纬度"
        :return: 距离，单位为米
        """

        if not cls.is_init:
            cls.init()

        # 使用字典快速查找
        key = (origin, destination)
        if key in cls.driving_distance_cache:
            return cls.driving_distance_cache[key]

        # 请求参数
        request_params = {
            "key": cls.API_KEY,
            "origin": origin,
            "destination": destination,
            "extensions": "base",  # 返回基本信息
            "strategy": 0  # 速度优先策略
        }

        try:
            # 发送请求
            resp = requests.get(cls.API_DRIVING_URL, params=request_params)
            resp.raise_for_status()  # 检查请求是否成功

            # 解析返回参数
            result = resp.json()

            if result.get("status") == "1" and result.get("info") == "OK":
                # 获取路径规划结果
                route = result.get("route", {})
                paths = route.get("paths", [])

                if paths:
                    # 获取第一条路径的距离
                    distance = int(paths[0].get("distance", 0))
                    # 保存数据
                    new_row = pd.DataFrame({
                        "origin": [origin],
                        "destination": [destination],
                        "distance": [distance]
                    })
                    cls.driving_distance_df = pd.concat([cls.driving_distance_df, new_row], ignore_index=True)

                    # 更新缓存
                    cls.driving_distance_cache[key] = distance

                    return distance

        except requests.exceptions.RequestException as e:
            print(f"请求异常: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            return None
        except Exception as e:
            print(f"发生错误: {e}")
            return None

    @classmethod
    def get_geocode(cls, address):
        """
        获取地址的经纬度地址

        :param address: 地址名称
        :return: 经度,纬度
        """

        if not cls.is_init:
            cls.init()

        # 使用字典快速查找
        if address in cls.geocode_cache:
            return cls.geocode_cache[address]

        # 请求参数
        request_params = {
            "key": cls.API_KEY,
            "address": address,
            "output": "json"
        }

        try:
            # 发送请求
            resp = requests.get(cls.API_GEO_URL, params=request_params)
            resp.raise_for_status()

            # 解析返回参数
            result = resp.json()

            if result.get("status") == "1" and result.get("info") == "OK":
                # 获取地理编码结果
                geocodes = result.get("geocodes", [])

                if geocodes:
                    # 获取第一个结果的坐标
                    location = geocodes[0].get("location")
                    if location:
                        lng, lat = map(float, location.split(","))
                        # 保存数据
                        new_row = pd.DataFrame({
                            "address": [address],
                            "lng,lat": [f"{lng},{lat}"]
                        })
                        cls.geocode_df = pd.concat([cls.geocode_df, new_row], ignore_index=True)

                        # 更新缓存
                        cls.geocode_cache[address] = f"{lng},{lat}"

                        return f"{lng},{lat}"

        except requests.exceptions.RequestException as e:
            print(f"请求异常: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            return None
        except Exception as e:
            print(f"发生错误: {e}")
            return None
