"""
标题和描述生成器
使用提示词工程或DPO模型生成吸引人的文案
"""

# 文案风格模板
TITLE_TEMPLATES = {
    '新疆': {
        'prefix': ['天山南北', '大美西域', '丝路明珠', '北疆秘境', '南疆风情'],
        'style': '｜{spots}，{theme}'
    },
    '西藏': {
        'prefix': ['雪域高原', '天路朝圣', '藏地密码', '圣域之旅', '云端西藏'],
        'style': '｜{spots}，{theme}'
    },
    '云南': {
        'prefix': ['彩云之南', '滇西秘境', '茶马古道', '高原明珠', '风花雪月'],
        'style': '｜{spots}，{theme}'
    },
    '四川': {
        'prefix': ['天府之国', '川西秘境', '蜀山之王', '人间仙境', '九寨天堂'],
        'style': '｜{spots}，{theme}'
    },
    '甘肃': {
        'prefix': ['河西走廊', '丝路重镇', '陇原风光', '敦煌印象', '祁连秘境'],
        'style': '｜{spots}，{theme}'
    },
    '青海': {
        'prefix': ['高原净土', '青海湖畔', '柴达木梦', '三江源头', '天空之境'],
        'style': '｜{spots}，{theme}'
    },
    '宁夏': {
        'prefix': ['塞上江南', '黄河金岸', '西夏风情', '沙湖明珠', '贺兰山下'],
        'style': '｜{spots}，{theme}'
    },
    '内蒙古': {
        'prefix': ['草原天路', '大漠孤烟', '蒙古风情', '呼伦贝尔', '阿尔山秘境'],
        'style': '｜{spots}，{theme}'
    }
}

THEME_KEYWORDS = {
    '雪山': ['触摸冰川', '仰望雪峰', '雪域奇观'],
    '湖泊': ['碧波荡漾', '镜面天空', '高原明珠'],
    '草原': ['策马奔腾', '风吹草低', '牧歌悠扬'],
    '古城': ['穿越时空', '寻古探今', '历史回响'],
    '峡谷': ['地质奇观', '峡谷探秘', '鬼斧神工'],
    '沙漠': ['大漠风光', '沙海奇观', '丝路驼铃']
}

def generate_title(route_pois, province, query=None):
    """
    生成路线标题
    
    Args:
        route_pois: POI列表
        province: 省份
        query: 用户查询（可选）
    
    Returns:
        str: 生成的标题
    """
    import random
    
    # 获取省份模板
    template = TITLE_TEMPLATES.get(province, TITLE_TEMPLATES['新疆'])
    
    # 选择3个代表性景点
    if len(route_pois) <= 3:
        spots_str = '-'.join([p['poi_name'] for p in route_pois])
    else:
        spots_str = f"{route_pois[0]['poi_name']}-{route_pois[1]['poi_name']}-{route_pois[2]['poi_name']}"
    
    # 根据景点特征选择主题词
    theme = _extract_theme_from_pois(route_pois)
    
    # 生成标题
    prefix = random.choice(template['prefix'])
    title = f"{prefix}{template['style'].format(spots=spots_str, theme=theme)}"
    
    return title

def generate_description(route_pois, province, total_hours, query=None):
    """
    生成路线描述
    
    Args:
        route_pois: POI列表
        province: 省份
        total_hours: 总行程时间
        query: 用户查询（可选）
    
    Returns:
        str: 生成的描述
    """
    num_pois = len(route_pois) - 2  # 减去起终点
    
    # 提取关键景点
    highlights = []
    for poi in route_pois[1:-1]:  # 排除起终点
        if '湖' in poi['poi_name'] or '山' in poi['poi_name'] or '草原' in poi['poi_name']:
            highlights.append(poi['poi_name'])
            if len(highlights) >= 3:
                break
    
    # 构建描述
    desc_parts = []
    
    if query:
        desc_parts.append(f"根据您的需求「{query}」")
    
    desc_parts.append(f"为您精心规划了这条{province}深度游路线。")
    desc_parts.append(f"行程涵盖{num_pois}个精选景点，预计用时{total_hours:.1f}小时。")
    
    if highlights:
        highlights_str = '、'.join(highlights)
        desc_parts.append(f"沿途将游览{highlights_str}等知名景点。")
    
    desc_parts.append(f"让您在有限的时间内，领略{province}最精华的风景，体验最地道的文化。")
    
    return "".join(desc_parts)

def _extract_theme_from_pois(route_pois):
    """从POI列表中提取主题词"""
    import random
    
    # 统计景点类型
    poi_names = ' '.join([p['poi_name'] for p in route_pois])
    
    # 匹配主题
    matched_themes = []
    for keyword, themes in THEME_KEYWORDS.items():
        if keyword in poi_names:
            matched_themes.extend(themes)
    
    if matched_themes:
        return random.choice(matched_themes)
    else:
        return '探索未知之美'

if __name__ == "__main__":
    # 测试
    test_pois = [
        {'poi_name': '喀纳斯湖', 'poi_city': '阿勒泰'},
        {'poi_name': '禾木村', 'poi_city': '阿勒泰'},
        {'poi_name': '白哈巴村', 'poi_city': '阿勒泰'}
    ]
    
    title = generate_title(test_pois, '新疆')
    desc = generate_description(test_pois, '新疆', 8.5, '想去喀纳斯看秋天的景色')
    
    print(f"标题: {title}")
    print(f"描述: {desc}")

