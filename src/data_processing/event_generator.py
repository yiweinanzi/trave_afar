"""
准备数据：从原始景点Excel转换为标准格式的 poi.csv 和模拟的 user_events.csv
"""
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# 新疆各区域中心大致坐标（用于生成模拟经纬度）
REGION_COORDS = {
    '乌鲁木齐': (87.6168, 43.8256),
    '阿勒泰': (88.1396, 47.8484),
    '塔城': (82.9856, 46.7456),
    '伊犁': (81.3179, 43.9168),
    '哈密': (93.5156, 42.8330),
    '吐鲁番': (89.1841, 42.9476),
    '巴音郭楞': (86.1509, 41.7686),
    '阿库': (80.2651, 41.1707),  # 阿克苏
    '和田': (79.9224, 37.1167),
    '喀什': (75.9891, 39.4677),
    '跨区域': (87.6168, 43.8256),  # 默认乌鲁木齐
}

def prepare_poi_data(source_path='data/poi_source.xlsx', output_path='data/poi.csv'):
    """
    将原始景点数据转换为标准格式
    poi_id,name,lat,lon,open_min,close_min,stay_min,city,description,landscapes,activities
    """
    df = pd.read_excel(source_path)
    
    poi_data = []
    for idx, row in df.iterrows():
        name = row['景点名称']
        city = row['地区']
        desc = str(row['景点介绍']) if pd.notna(row['景点介绍']) else ""
        landscapes = str(row['涉及景色']) if pd.notna(row['涉及景色']) else ""
        activities = str(row['涉及活动']) if pd.notna(row['涉及活动']) else ""
        
        # 获取区域中心坐标
        base_lon, base_lat = REGION_COORDS.get(city, (87.6168, 43.8256))
        
        # 在区域中心附近随机偏移（±0.5度，约50公里范围）
        lon = base_lon + random.uniform(-0.5, 0.5)
        lat = base_lat + random.uniform(-0.5, 0.5)
        
        # 模拟开放时间（分钟数，从午夜开始）
        # 大部分景点 9:00-19:00 开放
        open_min = random.choice([480, 540, 600])  # 8:00, 9:00, 10:00
        close_min = random.choice([1080, 1140, 1200])  # 18:00, 19:00, 20:00
        
        # 全天开放的景点（道路、市区等）
        if any(keyword in name for keyword in ['公路', '市区', '机场', '方向']):
            open_min = 0
            close_min = 1440  # 24小时
        
        # 模拟停留时长（分钟）
        if '市区' in name or '机场' in name:
            stay_min = 60  # 1小时
        elif any(keyword in name for keyword in ['公路', '方向', '沙漠公路']):
            stay_min = 120  # 2小时（路上时间）
        elif any(keyword in name for keyword in ['湖', '山', '峡谷', '草原']):
            stay_min = random.choice([120, 180, 240])  # 2-4小时
        else:
            stay_min = random.choice([90, 120, 150])  # 1.5-2.5小时
        
        poi_data.append({
            'poi_id': f'POI_{idx:04d}',
            'name': name,
            'lat': round(lat, 6),
            'lon': round(lon, 6),
            'open_min': open_min,
            'close_min': close_min,
            'stay_min': stay_min,
            'city': city,
            'description': desc[:200] if desc else "",  # 限制描述长度
            'landscapes': landscapes,
            'activities': activities
        })
    
    poi_df = pd.DataFrame(poi_data)
    poi_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✓ 生成 POI 数据: {output_path}, 共 {len(poi_df)} 个景点")
    return poi_df

def generate_user_events(poi_df, output_path='data/user_events.csv', 
                         num_users=500, events_per_user_range=(5, 30)):
    """
    生成模拟的用户行为数据
    user_id,poi_id,timestamp,action (click|fav|visit)
    """
    events = []
    poi_ids = poi_df['poi_id'].tolist()
    
    # 生成时间戳（最近6个月的数据）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    for user_id in range(1, num_users + 1):
        # 每个用户随机生成 5-30 个事件
        num_events = random.randint(*events_per_user_range)
        
        # 为每个用户选择偏好区域（模拟用户偏好）
        preferred_cities = random.sample(list(REGION_COORDS.keys()), k=random.randint(2, 4))
        preferred_pois = poi_df[poi_df['city'].isin(preferred_cities)]['poi_id'].tolist()
        
        # 70% 从偏好区域选择，30% 随机探索
        for _ in range(num_events):
            if random.random() < 0.7 and preferred_pois:
                poi_id = random.choice(preferred_pois)
            else:
                poi_id = random.choice(poi_ids)
            
            # 随机时间戳
            random_days = random.randint(0, 180)
            timestamp = int((start_date + timedelta(days=random_days)).timestamp())
            
            # 行为类型：60% click, 25% fav, 15% visit
            action = random.choices(['click', 'fav', 'visit'], weights=[0.6, 0.25, 0.15])[0]
            
            events.append({
                'user_id': f'U{user_id:04d}',
                'poi_id': poi_id,
                'timestamp': timestamp,
                'action': action
            })
    
    events_df = pd.DataFrame(events)
    events_df = events_df.sort_values(['user_id', 'timestamp'])
    events_df.to_csv(output_path, index=False)
    print(f"✓ 生成用户事件数据: {output_path}, 共 {len(events_df)} 条记录")
    return events_df

if __name__ == "__main__":
    import os
    os.makedirs('data', exist_ok=True)
    
    # 准备 POI 数据
    poi_df = prepare_poi_data()
    print(f"\nPOI 数据样例：")
    print(poi_df.head(3))
    
    # 生成用户事件数据
    events_df = generate_user_events(poi_df)
    print(f"\n用户事件数据样例：")
    print(events_df.head(10))
    
    print("\n✓ 数据准备完成！")

