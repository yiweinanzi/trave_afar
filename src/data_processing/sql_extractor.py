"""
从SQL文件中提取所有省份的真实景点数据
支持：新疆、西藏、云南、四川、甘肃、宁夏、内蒙古、青海
"""
import pandas as pd
import re
import os

# 根据ID前缀确定省份
PROVINCE_BY_ID = {
    '001': '新疆',
    '002': '西藏',
    '003': '云南',
    '004': '四川',
    '005': '甘肃',
    '006': '宁夏',
    '007': '内蒙古',
    '008': '青海'
}

def parse_time_to_minutes(time_str):
    """将时间字符串转换为从午夜开始的分钟数"""
    if not time_str or time_str.strip() == '' or '0:00-23:59' in time_str or '暂无' in time_str or '具体' in time_str:
        return 0, 1440  # 全天开放
    
    # 提取时间格式 HH:MM-HH:MM
    match = re.search(r'(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})', time_str)
    if match:
        open_h, open_m, close_h, close_m = map(int, match.groups())
        return open_h * 60 + open_m, close_h * 60 + close_m
    
    # 如果只有单个时间
    match = re.search(r'(\d{1,2}):(\d{2})', time_str)
    if match:
        h, m = map(int, match.groups())
        return h * 60 + m, (h + 10) * 60 + m  # 默认开放10小时
    
    # 默认9:00-19:00
    return 540, 1140

def get_province_by_id(poi_id):
    """根据ID前缀获取省份"""
    prefix = poi_id[:3]
    return PROVINCE_BY_ID.get(prefix, '未知')

def extract_city_from_name_and_desc(name, desc, province):
    """从景点名称和描述中提取城市"""
    # 新疆城市关键词
    xinjiang_cities = {
        '乌鲁木齐': ['乌鲁木齐', '乌市'],
        '阿勒泰': ['阿勒泰', '喀纳斯', '禾木', '布尔津', '白哈巴', '可可托海', '富蕴', '福海', '哈巴河'],
        '塔城': ['塔城', '奎屯', '克拉玛依', '独山子', '安集海', '世界魔鬼城'],
        '伊犁': ['伊犁', '伊宁', '那拉提', '赛里木湖', '昭苏', '特克斯', '巩留', '夏塔', '唐布拉', '果子沟', '霍城'],
        '哈密': ['哈密'],
        '吐鲁番': ['吐鲁番', '火焰山', '葡萄沟', '鄯善', '交河', '高昌'],
        '巴音郭楞': ['巴音郭楞', '巴音布鲁克', '库尔勒', '博湖', '轮台', '焉耆', '和静', '和硕'],
        '阿克苏': ['阿克苏', '库车', '温宿', '拜城', '新和', '沙雅', '阿瓦提'],
        '和田': ['和田', '洛浦', '策勒', '于田', '民丰'],
        '喀什': ['喀什', '帕米尔', '塔什库尔干', '慕士塔格', '莎车', '叶城', '泽普', '巴楚', '英吉沙', '麦盖提', '岳普湖'],
        '克州': ['克州', '阿图什', '乌恰', '阿克陶', '阿合奇'],
        '博州': ['博州', '博乐', '精河', '温泉', '阿拉山口'],
        '昌吉': ['昌吉', '呼图壁', '玛纳斯', '吉木萨尔', '奇台', '木垒'],
        '石河子': ['石河子'],
    }
    
    # 其他省份城市关键词（简化版）
    other_cities = {
        '拉萨': ['拉萨', '布达拉宫', '大昭寺', '八廓街'],
        '日喀则': ['日喀则', '扎什伦布寺', '珠峰', '定日', '萨迦'],
        '林芝': ['林芝', '波密', '米林', '巴松措', '鲁朗'],
        '昆明': ['昆明', '滇池'],
        '大理': ['大理', '洱海', '古城'],
        '丽江': ['丽江', '玉龙雪山', '束河'],
        '成都': ['成都'],
        '兰州': ['兰州'],
        '西宁': ['西宁'],
    }
    
    # 合并所有城市关键词
    all_cities = {}
    if province == '新疆':
        all_cities = xinjiang_cities
    else:
        all_cities = other_cities
    
    # 检查景点名称
    for city, keywords in all_cities.items():
        if any(kw in name for kw in keywords):
            return city
    
    # 检查描述
    if desc:
        for city, keywords in all_cities.items():
            if any(kw in desc[:200] for kw in keywords):
                return city
    
    # 如果找不到，返回省份作为城市
    return province

def estimate_stay_time(name, desc):
    """根据景点类型估计停留时长（分钟）"""
    if '市区' in name or '机场' in name or '站' in name:
        return 60
    elif any(kw in name for kw in ['公路', '方向', '高速', '道路', 'S21', 'S101', 'G']):
        return 120
    elif any(kw in name for kw in ['博物馆', '清真寺', '古城', '大巴扎', '寺', '庙', '宫']):
        return 90
    elif any(kw in name for kw in ['湖', '草原', '峡谷', '公园', '雪山', '冰川', '山口', '瀑布']):
        return 180
    elif '村' in name or '镇' in name or '乡' in name:
        return 120
    elif '滑雪' in name or '温泉' in name:
        return 240
    else:
        return 150

def parse_go_address_sql(sql_file='sql/go_address.sql', output_csv='data/poi.csv'):
    """解析 go_address.sql 文件，提取所有省份数据"""
    with open(sql_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取所有 INSERT 语句
    pattern = r"INSERT INTO `go_address` VALUES \('([^']+)', '([^']+)', '([^']*)', '([^']*)', '([^']*)', '([^']*)', NULL\);"
    matches = re.findall(pattern, content, re.DOTALL)
    
    print(f"找到 {len(matches)} 个景点记录")
    
    poi_data = []
    for idx, match in enumerate(matches):
        poi_id, name, time_str, description, airport, coords = match
        
        # 根据ID确定省份
        province = get_province_by_id(poi_id)
        
        # 解析坐标
        if coords and ',' in coords:
            try:
                parts = coords.split(',')
                lon = float(parts[0])
                lat = float(parts[1])
            except:
                # 如果解析失败，使用默认坐标（乌鲁木齐）
                lon, lat = 87.6168, 43.8256
        else:
            # 默认坐标
            lon, lat = 87.6168, 43.8256
        
        # 解析营业时间
        open_min, close_min = parse_time_to_minutes(time_str)
        
        # 提取城市
        city = extract_city_from_name_and_desc(name, description, province)
        
        # 估计停留时长
        stay_min = estimate_stay_time(name, description)
        
        # 清理描述文本
        clean_desc = description.replace('\n', ' ').replace('\r', ' ').strip()
        
        poi_data.append({
            'poi_id': poi_id,
            'name': name,
            'lat': round(lat, 6),
            'lon': round(lon, 6),
            'open_min': open_min,
            'close_min': close_min,
            'stay_min': stay_min,
            'province': province,
            'city': city,
            'description': clean_desc[:500],  # 限制长度
            'time_str': time_str,
            'airport': airport,
        })
    
    # 创建DataFrame
    df = pd.DataFrame(poi_data)
    
    # 保存
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    
    print(f"\n✓ 成功导出 {len(df)} 个景点到 {output_csv}")
    
    print(f"\n省份分布:")
    print(df['province'].value_counts())
    
    print(f"\n各省份景点数量:")
    for prov in sorted(df['province'].unique()):
        count = len(df[df['province'] == prov])
        print(f"  {prov}: {count} 个景点")
    
    # 显示每个省份的示例
    print(f"\n各省份示例景点:")
    for prov in sorted(df['province'].unique()):
        prov_df = df[df['province'] == prov]
        print(f"\n【{prov}】")
        sample = prov_df.head(3)[['poi_id', 'name', 'city', 'lat', 'lon']]
        print(sample.to_string(index=False))
    
    return df

if __name__ == "__main__":
    df = parse_go_address_sql()
    print("\n✓ 数据提取完成！")
    print(f"\n总计: {len(df)} 个景点，覆盖 {df['province'].nunique()} 个省份")
