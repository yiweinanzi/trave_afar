"""
时间矩阵构建器
基于Haversine距离或OSMnx路网计算POI间的旅行时间
"""
import numpy as np
import pandas as pd
import os
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    计算两点间的 Haversine 距离（公里）
    
    Args:
        lat1, lon1: 起点经纬度
        lat2, lon2: 终点经纬度
    
    Returns:
        float: 距离（公里）
    """
    R = 6371  # 地球半径（公里）
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c

def build_time_matrix(poi_csv='data/poi.csv',
                     output_path='outputs/routing/time_matrix.npy',
                     avg_speed_kmh=60,
                     poi_ids=None):
    """
    构建时间矩阵（秒）
    
    Args:
        poi_csv: POI数据文件
        output_path: 输出文件路径
        avg_speed_kmh: 平均行驶速度（公里/小时）
        poi_ids: 指定POI ID列表（如果为None，使用全部）
    
    Returns:
        tuple: (time_matrix, poi_df)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 读取POI数据
    df = pd.read_csv(poi_csv)
    
    # 筛选指定的POI
    if poi_ids is not None:
        df = df[df['poi_id'].isin(poi_ids)].reset_index(drop=True)
    
    n = len(df)
    print(f"构建时间矩阵: {n}x{n}")
    print(f"平均速度: {avg_speed_kmh} km/h")
    
    # 初始化时间矩阵（秒）
    T = np.zeros((n, n), dtype=np.int32)
    
    for i in range(n):
        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{n}")
        
        lat1, lon1 = df.iloc[i]['lat'], df.iloc[i]['lon']
        
        for j in range(n):
            if i == j:
                continue
            
            lat2, lon2 = df.iloc[j]['lat'], df.iloc[j]['lon']
            
            # 计算 Haversine 距离（公里）
            dist_km = haversine_distance(lat1, lon1, lat2, lon2)
            
            # 转换为行驶时间（秒）
            time_hours = dist_km / avg_speed_kmh
            T[i, j] = int(time_hours * 3600)
    
    # 保存时间矩阵
    np.save(output_path, T)
    
    print(f"\n✓ 时间矩阵保存到: {output_path}")
    print(f"  矩阵维度: {T.shape}")
    print(f"  平均行程时间: {T[T > 0].mean() / 60:.2f} 分钟")
    print(f"  最大行程时间: {T.max() / 60:.2f} 分钟")
    print(f"  最小行程时间: {T[T > 0].min() / 60:.2f} 分钟")
    
    return T, df

if __name__ == "__main__":
    # 测试
    T, df = build_time_matrix()
    print("\n示例行程时间（前5个POI）:")
    for i in range(min(5, len(df))):
        for j in range(min(5, len(df))):
            if i != j:
                print(f"  {df.iloc[i]['name'][:15]} -> {df.iloc[j]['name'][:15]}: {T[i,j]/60:.1f}分钟")

