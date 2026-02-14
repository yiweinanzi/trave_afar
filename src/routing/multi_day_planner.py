"""
多日行程规划器
支持2-13天的旅游路线规划，参考历史路线数据
"""
import pandas as pd
import numpy as np
from .vrptw_solver import VRPTWSolver
from .time_matrix_builder import build_time_matrix
import json

class MultiDayPlanner:
    """多日行程规划器"""
    
    def __init__(self):
        """初始化多日规划器"""
        self.historical_routes = self._load_historical_routes()
    
    def _load_historical_routes(self):
        """加载历史路线数据"""
        import os
        if os.path.exists('data/historical_routes.csv'):
            return pd.read_csv('data/historical_routes.csv')
        else:
            print("⚠️ 未找到历史路线数据")
            return None
    
    def plan_multi_day(self, candidate_pois, days, hours_per_day=8, start_city=None):
        """
        规划多日行程
        
        Args:
            candidate_pois: 候选POI DataFrame
            days: 天数
            hours_per_day: 每天游玩时长
            start_city: 起点城市
        
        Returns:
            dict: 多日行程规划结果
        """
        print(f"\n=== 多日行程规划 ===")
        print(f"天数: {days}天")
        print(f"每天时长: {hours_per_day}小时")
        print(f"候选POI: {len(candidate_pois)}个")
        
        # 按城市分组POI
        city_groups = candidate_pois.groupby('city')
        
        print(f"\n城市分布:")
        for city, group in city_groups:
            print(f"  {city}: {len(group)}个")
        
        # 为每天分配POI
        daily_plans = []
        remaining_pois = candidate_pois.copy()
        
        for day in range(1, days + 1):
            print(f"\n规划第{day}天...")
            
            # 选择当天的POI（优先同城市）
            if day == 1 and start_city:
                # 第一天从起点城市开始
                day_pois = remaining_pois[remaining_pois['city'] == start_city].head(10)
            else:
                # 选择评分最高的POI所在城市
                top_city = remaining_pois.iloc[0]['city'] if len(remaining_pois) > 0 else None
                if top_city:
                    day_pois = remaining_pois[remaining_pois['city'] == top_city].head(10)
                else:
                    day_pois = remaining_pois.head(10)
            
            if len(day_pois) == 0:
                print(f"  第{day}天：无可用POI")
                break
            
            # 构建时间矩阵并规划路线
            time_matrix, poi_df = build_time_matrix(poi_ids=day_pois['poi_id'].tolist())
            
            solver = VRPTWSolver(poi_df, time_matrix)
            solution = solver.solve(
                depot_index=0,
                max_duration_hours=hours_per_day,
                time_limit_seconds=20
            )
            
            if solution:
                daily_plans.append({
                    'day': day,
                    'city': day_pois.iloc[0]['city'],
                    'route': solution['routes'][0],
                    'hours': solution['total_hours'],
                    'num_pois': solution['visited_pois']
                })
                
                print(f"  ✓ 第{day}天规划完成: {solution['visited_pois']}个景点, {solution['total_hours']:.1f}小时")
                
                # 移除已访问的POI
                visited_poi_ids = [p['poi_id'] for p in solution['routes'][0][1:-1]]
                remaining_pois = remaining_pois[~remaining_pois['poi_id'].isin(visited_poi_ids)]
            else:
                print(f"  ✗ 第{day}天：未找到可行路线")
                break
        
        result = {
            'days': len(daily_plans),
            'daily_plans': daily_plans,
            'total_pois': sum(p['num_pois'] for p in daily_plans),
            'total_hours': sum(p['hours'] for p in daily_plans)
        }
        
        print(f"\n✓ {len(daily_plans)}天行程规划完成")
        print(f"  总景点数: {result['total_pois']}")
        print(f"  总时长: {result['total_hours']:.1f}小时")
        
        return result
    
    def recommend_similar_historical_routes(self, user_intent, topk=5):
        """
        推荐相似的历史路线
        
        Args:
            user_intent: 用户意图
            topk: 返回Top-K条路线
        
        Returns:
            DataFrame: 相似的历史路线
        """
        if self.historical_routes is None:
            return None
        
        routes = self.historical_routes.copy()
        
        # 根据天数过滤
        if user_intent.get('duration_days'):
            target_days = user_intent['duration_days']
            routes = routes[routes['days'] == target_days]
        
        print(f"\n找到 {len(routes)} 条{user_intent.get('duration_days', 'N')}天的历史路线")
        
        return routes.head(topk)

if __name__ == "__main__":
    # 测试多日规划
    import pandas as pd
    
    print("="*60)
    print("测试多日行程规划")
    print("="*60)
    
    # 构造测试候选
    test_candidates = pd.read_csv('data/poi.csv')
    xinjiang_pois = test_candidates[test_candidates['province'] == '新疆'].head(30)
    
    planner = MultiDayPlanner()
    
    # 规划3天行程
    result = planner.plan_multi_day(
        candidate_pois=xinjiang_pois,
        days=3,
        hours_per_day=8,
        start_city='乌鲁木齐'
    )
    
    # 显示结果
    print(f"\n{'='*60}")
    print("行程总览")
    print(f"{'='*60}")
    
    for day_plan in result['daily_plans']:
        print(f"\n第{day_plan['day']}天 - {day_plan['city']}")
        print(f"  时长: {day_plan['hours']:.1f}小时")
        print(f"  景点:")
        for stop in day_plan['route'][1:-1]:
            print(f"    - {stop['poi_name']}")

