#!/usr/bin/env python
"""
SHP省份POI数据统计脚本
"""
import sys
sys.path.insert(0, 'src')
from pathlib import Path
import struct
import pandas as pd

SHP_DIR = Path('data/external/shengfen')

def read_dbf_header(dbf_path):
    with open(dbf_path, 'rb') as f:
        f.seek(8)
        record_count = struct.unpack('<I', f.read(4))[0]
        return {'record_count': record_count}

province_names = {
    'anhui': '安徽', 'beijing': '北京', 'chongqing': '重庆',
    'fujian': '福建', 'gansu': '甘肃', 'guangdong': '广东',
    'guangxi': '广西', 'hainan': '海南', 'hebei': '河北',
    'heilongjiang': '黑龙江', 'henan': '河南', 'hubei': '湖北',
    'hunan': '湖南', 'inner-mongolia': '内蒙古', 'jiangsu': '江苏',
    'jilin': '吉林', 'liaoning': '辽宁', 'macau': '澳门',
    'ningxia': '宁夏', 'qinghai': '青海'
}

# 收集所有POI信息
all_pois = []

for province_dir in sorted(SHP_DIR.iterdir()):
    if province_dir.is_dir() and 'free' in province_dir.name.lower():
        province_key = province_dir.name.replace('-260213-free.shp', '').replace('-260212-free.shp', '')
        province = province_names.get(province_key, province_key)

        # 查找各种POI数据文件
        for dbf_file in province_dir.glob('*.dbf'):
            if 'poi' in dbf_file.name.lower() or 'place' in dbf_file.name.lower():
                header = read_dbf_header(dbf_file)
                count = header['record_count']

                all_pois.append({
                    'province': province,
                    'source_file': dbf_file.name,
                    'record_count': count,
                    'type': 'poi' if 'poi' in dbf_file.name.lower() else 'place'
                })

print(f'解析到 {len(all_pois)} 个POI数据文件')

# 按省份汇总
summary = {}
for poi in all_pois:
    prov = poi['province']
    if prov not in summary:
        summary[prov] = {'poi_files': [], 'total_records': 0}
    summary[prov]['poi_files'].append(poi)
    summary[prov]['total_records'] += poi['record_count']

# 生成汇总DataFrame
summary_data = []
for prov, data in sorted(summary.items()):
    # 取最大值作为POI数
    max_records = max(f['record_count'] for f in data['poi_files'])
    summary_data.append({
        'province': prov,
        'poi_files_count': len(data['poi_files']),
        'total_dbf_records': data['total_records'],
        'estimated_poi': max_records
    })

df = pd.DataFrame(summary_data)

print('\n=== 省份POI统计汇总 ===')
print(df.to_string(index=False))
print(f'\n总POI数: {df["estimated_poi"].sum():,}')

# 保存
df.to_csv('data/shengfen_poi_summary.csv', index=False)
print('\n保存到: data/shengfen_poi_summary.csv')
