#!/usr/bin/env python3
"""
从路线模板生成高质量SFT训练数据

��持三类任务:
1. 意图理解 - 从用户需求提取结构化信息
2. 路线生成 - 生成多日POI序列
3. 文案生成 - 生成路线标题和描述
"""
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd


# 省份到关键词映射
PROVINCE_KEYWORDS = {
    "新疆": ["沙漠", "草原", "雪山", "喀纳斯", "天池", "葡萄", "哈密瓜", "羊肉", "丝绸之路"],
    "西藏": ["布达拉宫", "雪山", "高原", "朝圣", "转山", "纳木错", "珠峰", "林芝", "雅鲁藏布"],
    "云南": ["古城", "丽江", "大理", "香格里拉", "洱海", "玉龙雪山", "西双版纳", "泼水节", "民族文化"],
    "四川": ["熊猫", "火锅", "成都", "九寨沟", "峨眉山", "乐山", "稻城", "亚丁", "川菜"],
    "北京": ["故宫", "长城", "天安门", "颐和园", "烤鸭", "胡同", "四合院", "天坛", "文化"],
    "上海": ["外滩", "东方明珠", "迪士尼", "南京路", "小吃", "现代", "繁华", "夜景"],
    "陕西": ["兵马俑", "西安", "古城", "美食", "历史", "大唐", "华清池", "华山", "肉夹馍"],
    "甘肃": ["敦煌", "莫高窟", "月牙泉", "嘉峪关", "丝绸之路", "张掖", "丹霞", "拉面"],
    "青海": ["青海湖", "茶卡盐湖", "塔尔寺", "油菜花", "高原", "天空之镜"],
    "贵州": ["黄果树", "瀑布", "千户苗寨", "茅台", "酸汤鱼", "溶洞", "峡谷"],
    "广西": ["桂林", "漓江", "阳朔", "山水", "米粉", "北海", "银滩", "德天瀑布"],
    "海南": ["三亚", "海滩", "热带", "海鲜", "度假", "椰林", "天涯海角", "蜈支洲"],
    "福建": ["厦门", "鼓浪屿", "土楼", "武夷山", "茶", "沙县小吃", "海滩"],
    "浙江": ["杭州", "西湖", "普陀", "乌镇", "古镇", "嘉兴", "茶", "丝绸"],
    "江苏": ["苏州", "园林", "周庄", "南京", "盐水鸭", "太湖", "扬州"],
    "湖南": ["张家界", "凤凰", "长沙", "臭豆腐", "辣椒", "岳阳楼", "韶山"],
    "湖北": ["武汉", "黄鹤楼", "三峡", "神农架", "热干面", "武当山"],
    "河南": ["少林寺", "洛阳", "牡丹", "龙门石窟", "开封", "烩面"],
    "山东": ["泰山", "青岛", "啤酒", "海鲜", "济南", "曲阜", "孔府", "蓬莱"],
    "山西": ["平遥", "古城", "乔家大院", "五台山", "云冈", "太原", "醋"],
    "河北": ["承德", "避暑山庄", "山海关", "北戴河", "雄安"],
    "内蒙古": ["草原", "沙漠", "骑马", "蒙古包", "烤全羊", "呼伦贝尔", "额济纳"],
    "辽宁": ["沈阳", "故宫", "大连", "海鲜", "千山"],
    "吉林": ["长白山", "雾凇", "滑雪", "长春", "电影"],
    "黑龙江": ["哈尔滨", "冰雪", "雪乡", "中央大街", "五大连池"],
    "安徽": ["黄山", "宏村", "西递", "九华山", "徽派", "臭鳜鱼"],
    "江西": ["庐山", "景德镇", "婺源", "滕王阁", "陶瓷"],
    "重庆": ["火锅", "山城", "洪崖洞", "大足", "小面", "夜景"],
    "天津": ["狗不理", "相声", "古文化街", "五大道", "海河"],
    "宁夏": ["沙坡头", "西夏", "贺兰山", "枸杞"],
    "澳门": ["赌城", "大三巴", "葡挞", "威尼斯人"],
    "香港": ["迪士尼", "维多利亚港", "购物", "美食", "太平山顶"],
}


def load_route_templates(templates_path: str) -> List[Dict]:
    """加载路线模板"""
    with open(templates_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_poi_data(poi_csv: str) -> pd.DataFrame:
    """加载POI数据"""
    return pd.read_csv(poi_csv, low_memory=False)


def generate_user_query(route: Dict, poi_df: pd.DataFrame) -> str:
    """根据路线生成用户查询"""
    province = route.get("province", "")
    days = route.get("days", 3)

    # 获取省份相关关键词
    keywords = PROVINCE_KEYWORDS.get(province, ["风景", "美食", "文化"])

    # 随机选择1-3个关键词
    selected_keywords = random.sample(keywords, min(3, len(keywords)))

    # 生成多种查询格式
    query_templates = [
        f"我想去{province}玩，喜欢{'、'.join(selected_keywords)}，计划{days}天",
        f"推荐一条{province}{days}日游，要包含{'、'.join(selected_keywords)}",
        f"{days}天{province}旅游，{'、'.join(selected_keywords)}主题",
        f"帮我规划{province}之旅，{days}天时间，偏好{'、'.join(selected_keywords)}",
        f"{'、'.join(selected_keywords)}向的{province}{days}天行程",
    ]

    return random.choice(query_templates)


def build_intent_understanding_samples(
    routes: List[Dict],
    poi_df: pd.DataFrame,
    num_samples: int = 500
) -> List[Dict]:
    """
    任务1: 意图理解
    输入: 用户自然语言查询
    输出: 结构化的旅行需求
    """
    samples = []

    for route in routes:
        if len(samples) >= num_samples:
            break

        province = route.get("province", "")
        days = route.get("days", 3)
        keywords = PROVINCE_KEYWORDS.get(province, [])

        # 生成用户查询
        query = generate_user_query(route, poi_df)

        # 构建结构化响应
        response = {
            "province": province,
            "duration_days": days,
            "interests": random.sample(keywords, min(3, len(keywords))) if keywords else [],
            "style": random.choice(["休闲游", "深度游", "摄影游", "文化游", "美食游", "探险游"]),
            "season": random.choice(["春", "夏", "秋", "冬", "全年"]),
            "budget_level": random.choice(["经济", "中等", "高端"]),
        }

        samples.append({
            "prompt": query,
            "response": json.dumps(response, ensure_ascii=False),
            "task_type": "intent_understanding"
        })

        # 为同一路线生成多个变体
        for _ in range(random.randint(1, 3)):
            if len(samples) >= num_samples:
                break
            samples.append({
                "prompt": generate_user_query(route, poi_df),
                "response": json.dumps(response, ensure_ascii=False),
                "task_type": "intent_understanding"
            })

    return samples


def build_route_generation_samples(
    routes: List[Dict],
    num_samples: int = 300
) -> List[Dict]:
    """
    任务2: 路线生成
    输入: 用户需求描述
    输出: 多日POI序列
    """
    samples = []

    for route in routes:
        if len(samples) >= num_samples:
            break

        province = route.get("province", "")
        days = route.get("days", 3)
        daily_routes = route.get("daily_routes", [])

        # 构建查询
        keywords = PROVINCE_KEYWORDS.get(province, [])
        selected = random.sample(keywords, min(2, len(keywords))) if keywords else ["风景"]

        query = f"推荐{province}{days}天行程，偏好{'、'.join(selected)}"

        # 构建响应 - 按天组织POI
        response = {
            "province": province,
            "days": days,
            "daily_pois": daily_routes,
            "total_poi_count": route.get("poi_count", 0)
        }

        samples.append({
            "prompt": query,
            "response": json.dumps(response, ensure_ascii=False),
            "task_type": "route_generation"
        })

    return samples


def build_content_generation_samples(
    routes: List[Dict],
    num_samples: int = 500
) -> List[Dict]:
    """
    任务3: 文案生成
    输入: 行程描述
    输出: 吸引人的标题和描述
    """
    samples = []

    # 标题模板
    title_templates = [
        "{province}秘境｜{poi_highlight}，{duration}日{theme}之旅",
        "{theme}{province}｜{poi_highlight}深度体验",
        "{duration}天{province}：{poi_highlight}，{highlight}",
        "{province}{theme}游｜{poi_highlight}精华路线",
        "发现{province}｜{poi_highlight}，{duration}天不卡点",
    ]

    # 主题词
    themes = [
        "自然风光", "历史文化", "美食探店", "摄影打卡",
        "休闲度假", "户外探险", "民俗体验", "亲子时光"
    ]

    # 高光词
    highlights = [
        "不留遗憾", "深度体验", "精选路线", "全程舒适",
        "打卡胜地", "网红景点", "小众秘境", "经典必游"
    ]

    for route in routes:
        if len(samples) >= num_samples:
            break

        province = route.get("province", "")
        days = route.get("days", 3)
        raw_route = route.get("raw_route", "")

        # 提取部分POI作为亮点
        daily_routes = route.get("daily_routes", [])
        all_pois = []
        for day_route in daily_routes:
            all_pois.extend(day_route)

        # 选择前3个POI作为亮点
        poi_highlight = "、".join(all_pois[:3])

        # 生成标题
        title = random.choice(title_templates).format(
            province=province,
            duration=f"{days}",
            poi_highlight=poi_highlight,
            theme=random.choice(themes),
            highlight=random.choice(highlights)
        )

        # 生成描述
        desc = f"这是一条精心设计的{province}{days}日游路线，涵盖{poi_highlight}等经典景点。"
        desc += f"行程涵盖{len(daily_routes)}天，总计{route.get('poi_count', 0)}个景点。"
        desc += random.choice([
            "适合喜欢深度游的旅行者，让您充分感受当地文化魅力。",
            "行程宽松不赶路，适合家庭出游或朋友结伴。",
            "精选当地特色体验，让旅程更有意义。",
            "专业路线规划，让您省心省力畅游{province}。"
        ])

        query = f"给'{raw_route[:50]}...'这条{province}{days}日游写个吸引人的标题和描述"

        response = {
            "title": title,
            "description": desc,
            "tags": [province, random.choice(themes), f"{days}日游"]
        }

        samples.append({
            "prompt": query,
            "response": json.dumps(response, ensure_ascii=False),
            "task_type": "content_generation"
        })

        # 为同一路线生成多个版本
        for _ in range(random.randint(1, 2)):
            if len(samples) >= num_samples:
                break

            title = random.choice(title_templates).format(
                province=province,
                duration=f"{days}",
                poi_highlight=poi_highlight,
                theme=random.choice(themes),
                highlight=random.choice(highlights)
            )

            response = {
                "title": title,
                "description": desc,
                "tags": [province, random.choice(themes), f"{days}日游"]
            }

            samples.append({
                "prompt": query,
                "response": json.dumps(response, ensure_ascii=False),
                "task_type": "content_generation"
            })

    return samples


def merge_with_existing_sft(
    new_samples: List[Dict],
    existing_sft_path: str
) -> List[Dict]:
    """合并现有SFT数据"""
    all_samples = new_samples[:]

    if Path(existing_sft_path).exists():
        with open(existing_sft_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_samples.append(json.loads(line))
                    except:
                        pass

    return all_samples


def main():
    parser = argparse.ArgumentParser(description="从路线模板生成SFT训练数据")
    parser.add_argument("--templates", default="data/all/route_templates.json", help="路线模板JSON")
    parser.add_argument("--poi-csv", default="data/all/poi_expanded.csv", help="POI数据")
    parser.add_argument("--output", default="outputs/datasets/sft_data.jsonl", help="输出文件")
    parser.add_argument("--existing-sft", default="outputs/datasets/sft_planner_samples.jsonl", help="现有SFT数据")
    parser.add_argument("--intent-samples", type=int, default=500, help="意图理解样本数")
    parser.add_argument("--route-samples", type=int, default=200, help="路线生成样本数")
    parser.add_argument("--content-samples", type=int, default=500, help="文案生成样本数")
    args = parser.parse_args()

    print("加载路线模板...")
    routes = load_route_templates(args.templates)
    print(f"  路线数: {len(routes)}")

    print("加载POI数据...")
    poi_df = load_poi_data(args.poi_csv)
    print(f"  POI数: {len(poi_df)}")

    print("\n生成SFT训练样本...")

    print("  1. 意图理解样本...")
    intent_samples = build_intent_understanding_samples(
        routes, poi_df, args.intent_samples
    )
    print(f"     生成: {len(intent_samples)} 条")

    print("  2. 路线生成样本...")
    route_samples = build_route_generation_samples(
        routes, args.route_samples
    )
    print(f"     生成: {len(route_samples)} 条")

    print("  3. 文案生成样本...")
    content_samples = build_content_generation_samples(
        routes, args.content_samples
    )
    print(f"     生成: {len(content_samples)} 条")

    # 合并所有样本
    all_samples = intent_samples + route_samples + content_samples

    # 合并现有SFT数据
    print(f"\n合并现有SFT数据: {args.existing_sft}")
    all_samples = merge_with_existing_sft(all_samples, args.existing_sft)

    # 打乱
    random.shuffle(all_samples)

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open('w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"\n✅ SFT数据生成完成!")
    print(f"  总样本数: {len(all_samples)}")
    print(f"  输出文件: {output_path}")
    print(f"\n  分布:")
    print(f"    意图理解: {len(intent_samples)}")
    print(f"    路线生成: {len(route_samples)}")
    print(f"    文案生成: {len(content_samples)}")
    print(f"    现有数据: {len(all_samples) - len(intent_samples) - len(route_samples) - len(content_samples)}")


if __name__ == "__main__":
    main()
