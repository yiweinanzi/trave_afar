"""
VRPTW (Vehicle Routing Problem with Time Windows) 求解器
参考: or-tools/examples/python/cvrptw_plot.py
"""
import numpy as np
import pandas as pd
import json
import os
from datetime import timedelta
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

class VRPTWSolver:
    """VRPTW求解器类"""
    
    def __init__(self, poi_df, time_matrix, start_time_min=480):
        """
        初始化求解器
        
        Args:
            poi_df: POI DataFrame
            time_matrix: 时间矩阵（秒）
            start_time_min: 出发时间（从午夜开始的分钟数，默认8:00）
        """
        self.poi_df = poi_df
        self.time_matrix = time_matrix
        self.start_time_min = start_time_min
        self.num_locations = len(poi_df)
        
        print(f"初始化 VRPTW 求解器:")
        print(f"  POI数量: {self.num_locations}")
        print(f"  出发时间: {start_time_min//60:02d}:{start_time_min%60:02d}")
    
    def solve(self, depot_index=0, max_duration_hours=10, 
              num_vehicles=1, time_limit_seconds=30):
        """
        求解VRPTW问题
        
        Args:
            depot_index: 起点/终点索引
            max_duration_hours: 最大行程时长（小时）
            num_vehicles: 车辆数量
            time_limit_seconds: 求解时间限制（秒）
        
        Returns:
            dict: 求解结果
        """
        print(f"\n开始求解 VRPTW...")
        print(f"  起点: {self.poi_df.iloc[depot_index]['name']}")
        print(f"  最大行程: {max_duration_hours} 小时")
        print(f"  车辆数: {num_vehicles}")
        
        # 创建路由模型
        manager = pywrapcp.RoutingIndexManager(
            self.num_locations,
            num_vehicles,
            depot_index
        )
        routing = pywrapcp.RoutingModel(manager)
        
        # 定义时间回调函数
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            
            # 行驶时间
            travel_time = int(self.time_matrix[from_node, to_node])
            
            # 停留时间（只在非起点处停留）
            if from_node != depot_index:
                service_time = int(self.poi_df.iloc[from_node]['stay_min'] * 60)
            else:
                service_time = 0
            
            return travel_time + service_time
        
        transit_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # 添加时间维度约束
        horizon = max_duration_hours * 3600  # 转换为秒
        routing.AddDimension(
            transit_callback_index,
            3600,          # 最大等待时间（1小时）
            horizon,       # 最大行程时长
            False,         # 不强制从0开始累计
            'Time'
        )
        time_dimension = routing.GetDimensionOrDie('Time')
        
        # 设置时间窗约束
        for i in range(self.num_locations):
            index = manager.NodeToIndex(i)
            
            # 获取POI的营业时间
            open_time = int(self.poi_df.iloc[i]['open_min'] * 60)
            close_time = int(self.poi_df.iloc[i]['close_min'] * 60)
            
            # 转换为相对于出发时间的时间
            start_time_sec = self.start_time_min * 60
            open_relative = max(0, open_time - start_time_sec)
            close_relative = close_time - start_time_sec
            
            # 全天开放的景点（如道路、市区）
            if close_time >= 1440 * 60:
                time_dimension.CumulVar(index).SetRange(0, horizon)
            else:
                # 确保时间窗口在合理范围内
                open_relative = max(0, open_relative)
                close_relative = min(horizon, close_relative)
                
                if open_relative < close_relative:
                    time_dimension.CumulVar(index).SetRange(
                        int(open_relative),
                        int(close_relative)
                    )
                else:
                    # 如果时间窗口无效，设置为全天
                    time_dimension.CumulVar(index).SetRange(0, horizon)
        
        # 允许跳过POI（如果时间窗不可达）
        penalty = 1000000  # 大罚分
        for i in range(self.num_locations):
            if i != depot_index:
                routing.AddDisjunction([manager.NodeToIndex(i)], penalty)
        
        # 设置搜索参数
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.FromSeconds(time_limit_seconds)
        
        # 求解
        print(f"正在求解（时间限制: {time_limit_seconds}秒）...")
        solution = routing.SolveWithParameters(search_parameters)
        
        # 解析结果
        if solution:
            result = self._parse_solution(
                routing, manager, solution, time_dimension, depot_index
            )
            print(f"✓ 求解成功！")
            return result
        else:
            print(f"✗ 未找到可行解")
            return None
    
    def _parse_solution(self, routing, manager, solution, time_dimension, depot_index):
        """解析求解结果"""
        routes = []
        total_time = 0
        
        for vehicle_id in range(routing.vehicles()):
            index = routing.Start(vehicle_id)
            route = []
            route_time = 0
            
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                time_var = time_dimension.CumulVar(index)
                arrival_time_sec = solution.Value(time_var)
                
                poi = self.poi_df.iloc[node]
                arrival_time_min = self.start_time_min + arrival_time_sec // 60
                
                route_info = {
                    'poi_id': poi['poi_id'],
                    'poi_name': poi['name'],
                    'poi_city': poi['city'],
                    'arrival_time_min': int(arrival_time_min),
                    'arrival_time_str': f"{arrival_time_min//60:02d}:{arrival_time_min%60:02d}",
                    'stay_min': int(poi['stay_min'])
                }
                route.append(route_info)
                
                index = solution.Value(routing.NextVar(index))
                route_time = arrival_time_sec
            
            # 添加终点
            node = manager.IndexToNode(index)
            time_var = time_dimension.CumulVar(index)
            arrival_time_sec = solution.Value(time_var)
            arrival_time_min = self.start_time_min + arrival_time_sec // 60
            
            route.append({
                'poi_id': self.poi_df.iloc[node]['poi_id'],
                'poi_name': self.poi_df.iloc[node]['name'],
                'poi_city': self.poi_df.iloc[node]['city'],
                'arrival_time_min': int(arrival_time_min),
                'arrival_time_str': f"{arrival_time_min//60:02d}:{arrival_time_min%60:02d}",
                'stay_min': 0
            })
            
            routes.append(route)
            total_time = max(total_time, route_time)
        
        # 打印路线
        print(f"\n路线详情:")
        for vid, route in enumerate(routes):
            if len(route) > 2:  # 排除只有起点终点的路线
                print(f"\n车辆 {vid + 1}:")
                for idx, stop in enumerate(route):
                    print(f"  {idx+1}. [{stop['arrival_time_str']}] {stop['poi_name']:<25} "
                          f"停留{stop['stay_min']}分钟")
        
        result = {
            'routes': routes,
            'total_time_hours': total_time / 3600,
            'total_hours': total_time / 3600,  # 兼容性
            'num_vehicles': len(routes),
            'objective_value': solution.ObjectiveValue(),
            'visited_pois': sum(len(r) - 2 for r in routes if len(r) > 2)  # 减去起终点
        }
        
        print(f"\n总行程时间: {result['total_hours']:.2f} 小时")
        print(f"访问景点数: {result['visited_pois']}")
        print(f"目标函数值: {result['objective_value']}")
        
        return result

if __name__ == "__main__":
    # 测试
    print("="*60)
    print("测试 VRPTW 求解器")
    print("="*60)
    
    # 加载数据
    poi_df = pd.read_csv('data/poi.csv').head(20)  # 测试用前20个
    
    # 构建时间矩阵
    time_matrix, poi_df = build_time_matrix(poi_ids=poi_df['poi_id'].tolist())
    
    # 求解
    solver = VRPTWSolver(poi_df, time_matrix)
    result = solver.solve(max_duration_hours=8)
    
    if result:
        # 保存结果
        with open('outputs/routing/test_result.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n✓ 结果保存到: outputs/routing/test_result.json")

