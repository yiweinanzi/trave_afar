#!/usr/bin/env python3
"""
从路线模板和用户行为生成DPO偏好对齐数据

策略:
1. 高质量路线模板 -> chosen
2. 扰动样本（打乱/替换POI） -> rejected
3. 用户参与度作为质量信号
"""
import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def load_route_templates(templates_path: str) -> List[Dict]:
    """加载路线模板"""
    with open(templates_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_poi_data(poi_csv: str) -> pd.DataFrame:
    """加载POI数据"""
    df = pd.read_csv(poi_csv, low_memory=False)
    # 确保有必要的列
    if 'name' not in df.columns:
        df['name'] = df.get('poi_id', '').astype(str)
    return df


def load_trajectories(trajectories_path: str) -> List[Dict]:
    """加载轨迹数据"""
    trajectories = []
    with open(trajectories_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    trajectories.append(json.loads(line))
                except:
                    pass
    return trajectories


def route_to_text(route: Dict, poi_df: pd.DataFrame) -> str:
    """将路线转换为文本描述"""
    daily_routes = route.get("daily_routes", [])

    parts = []
    for day_idx, day_pois in enumerate(daily_routes, 1):
        if isinstance(day_pois, list) and day_pois:
            # 尝试从POI数据中获取名称
            poi_names = []
            for poi in day_pois:
                if isinstance(poi, str):
                    # 查找POI名称
                    match = poi_df[poi_df['poi_id'] == poi]
                    if not match.empty:
                        name = match.iloc[0].get('name', poi)
                    else:
                        name = poi
                    poi_names.append(name)
                else:
                    poi_names.append(str(poi))

            parts.append(f"第{day_idx}天:{','.join(poi_names)}")

    return "；".join(parts)


def perturb_route(
    route_pois: List[str],
    all_pois: List[str],
    perturb_type: str = "shuffle"
) -> List[str]:
    """
    扰动路线生成rejected样本

    Args:
        route_pois: 原始POI列表
        all_pois: 所有POI池
        perturb_type: 扰动类型 - shuffle/replace/drop/duplicate
    """
    pois = route_pois[:]
    rng = random.Random()

    if perturb_type == "shuffle" and len(pois) > 2:
        # 打乱顺序（破坏地理合理性）
        rng.shuffle(pois[1:-1])  # 保持起点终点

    elif perturb_type == "replace" and len(pois) > 1:
        # 随机替换一个POI
        idx = rng.randint(1, len(pois) - 1)
        candidates = [p for p in all_pois if p not in pois]
        if candidates:
            pois[idx] = rng.choice(candidates)

    elif perturb_type == "drop" and len(pois) > 3:
        # 删除一个重要POI
        idx = rng.randint(1, len(pois) - 2)
        pois.pop(idx)

    elif perturb_type == "duplicate" and len(pois) > 1:
        # 重复添加一个POI（冗余）
        idx = rng.randint(0, len(pois) - 1)
        pois.insert(idx, pois[idx])

    return pois


def build_route_preference_pairs(
    routes: List[Dict],
    poi_df: pd.DataFrame,
    max_pairs: int = 200
) -> List[Dict]:
    """
    从路线模板构建偏好对
    """
    pairs = []
    all_poi_ids = poi_df['poi_id'].astype(str).tolist()[:10000]  # 限制POI池大小

    perturb_types = ["shuffle", "replace", "drop", "duplicate"]

    for route in routes:
        if len(pairs) >= max_pairs:
            break

        province = route.get("province", "")
        raw_route = route.get("raw_route", "")

        # 生成chosen描述
        chosen_text = route_to_text(route, poi_df)

        # 生成rejected
        perturb_type = random.choice(perturb_types)

        # 构造扰动路线
        daily_routes = route.get("daily_routes", [])
        if daily_routes:
            perturbed_route = {
                "daily_routes": [
                    perturb_route(day, all_poi_ids, perturb_type)
                    for day in daily_routes
                ],
                "province": province
            }
            rejected_text = route_to_text(perturbed_route, poi_df)
        else:
            rejected_text = f"随机推荐{province}景点"

        # 生成prompt
        prompts = [
            f"给{province}{route.get('days', 3)}日游设计行程",
            f"规划一条{province}旅游路线",
            f"{raw_route[:30]}... 这条路线怎么样？",
            f"推荐{province}的景点和路线",
        ]

        pairs.append({
            "prompt": random.choice(prompts),
            "chosen": chosen_text,
            "rejected": rejected_text,
            "source": "route_template"
        })

        # 为同一路线生成多个对
        for _ in range(random.randint(0, 2)):
            if len(pairs) >= max_pairs:
                break

            perturb_type = random.choice(perturb_types)
            perturbed_route = {
                "daily_routes": [
                    perturb_route(day, all_poi_ids, perturb_type)
                    for day in daily_routes
                ],
                "province": province
            }
            rejected_text = route_to_text(perturbed_route, poi_df)

            pairs.append({
                "prompt": random.choice(prompts),
                "chosen": chosen_text,
                "rejected": rejected_text,
                "source": "route_template"
            })

    return pairs


def build_title_preference_pairs(
    routes: List[Dict],
    max_pairs: int = 200
) -> List[Dict]:
    """
    构建文案标题偏好对
    """
    pairs = []

    # 高质量标题模板
    good_templates = [
        "{province}秘境｜{highlight}，{days}日{theme}",
        "{theme}{province}｜{highlight}深度体验",
        "发现{province}｜{highlight}，{days}天不卡点",
        "{province}{days}日：{highlight}精华路线",
    ]

    # 低质量标题模板
    bad_templates = [
        "{province}旅游路线推荐",
        "{province}{days}日游方案",
        "关于{province}的旅行计划",
        "{province}景点介绍",
    ]

    themes = ["自然风光", "历史文化", "美食探店", "摄影打卡", "休闲度假"]
    highlights = ["经典必游", "深度体验", "精选路线", "网红打卡"]

    for route in routes:
        if len(pairs) >= max_pairs:
            break

        province = route.get("province", "")
        days = route.get("days", 3)

        # 提取部分POI
        daily_routes = route.get("daily_routes", [])
        all_pois = []
        for day_route in daily_routes:
            all_pois.extend(day_route[:2])
        poi_str = "、".join(all_pois[:3]) if all_pois else "精选景点"

        # 生成chosen
        chosen = random.choice(good_templates).format(
            province=province,
            days=f"{days}",
            theme=random.choice(themes),
            highlight=poi_str
        )

        # 生成rejected
        rejected = random.choice(bad_templates).format(
            province=province,
            days=f"{days}"
        )

        prompts = [
            f"给{province}旅游路线写个吸引人的标题",
            f"为{province}{days}日游创作标题",
            f"写一个旅游路线标题",
        ]

        pairs.append({
            "prompt": random.choice(prompts),
            "chosen": chosen,
            "rejected": rejected,
            "source": "title_generation"
        })

    return pairs


def build_trajectory_preference_pairs(
    trajectories: List[Dict],
    poi_df: pd.DataFrame,
    max_pairs: int = 200
) -> List[Dict]:
    """
    从用户轨迹构建偏好对
    高参与度轨迹 -> chosen，低参与度/扰动 -> rejected
    """
    pairs = []

    # 按参与度排序
    sorted_traj = sorted(
        trajectories,
        key=lambda x: x.get("engagement_score", 0),
        reverse=True
    )

    # 高参与度作为chosen
    high_engagement = [t for t in sorted_traj if t.get("engagement_score", 0) >= 2.0]

    all_poi_ids = poi_df['poi_id'].astype(str).tolist()[:10000]

    for traj in high_engagement[:max_pairs]:
        poi_sequence = traj.get("poi_sequence", [])
        if len(poi_sequence) < 3:
            continue

        # 获取POI名称
        poi_names = []
        for poi_id in poi_sequence:
            match = poi_df[poi_df['poi_id'] == poi_id]
            if not match.empty:
                name = match.iloc[0].get('name', poi_id)
            else:
                name = poi_id
            poi_names.append(name)

        chosen_text = ",".join(poi_names)

        # 生成rejected - 扰动
        perturbed = perturb_route(poi_sequence, all_poi_ids, random.choice(["shuffle", "replace"]))
        perturbed_names = []
        for poi_id in perturbed:
            match = poi_df[poi_df['poi_id'] == poi_id]
            if not match.empty:
                name = match.iloc[0].get('name', poi_id)
            else:
                name = poi_id
            perturbed_names.append(name)
        rejected_text = ",".join(perturbed_names)

        pairs.append({
            "prompt": f"推荐一条旅游路线",
            "chosen": chosen_text,
            "rejected": rejected_text,
            "source": "user_trajectory"
        })

    return pairs


def save_dpo_data(pairs: List[Dict], output_path: str):
    """保存DPO数据为CSV格式"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "chosen", "rejected", "source"])
        writer.writeheader()
        writer.writerows(pairs)

    print(f"保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="从路线模板和用户行为构建DPO偏好数据")
    parser.add_argument("--templates", default="data/all/route_templates.json", help="路线模板")
    parser.add_argument("--trajectories", default="outputs/datasets/planner_trajectories.jsonl", help="轨迹数据")
    parser.add_argument("--poi-csv", default="data/all/poi_expanded.csv", help="POI数据")
    parser.add_argument("--output", default="outputs/datasets/dpo_prefs.csv", help="输出文件")
    parser.add_argument("--route-pairs", type=int, default=100, help="路线偏好对数量")
    parser.add_argument("--title-pairs", type=int, default=100, help="标题偏好对数量")
    parser.add_argument("--traj-pairs", type=int, default=100, help="轨迹偏好对数量")
    args = parser.parse_args()

    print("加载数据...")

    routes = load_route_templates(args.templates)
    print(f"  路线模板: {len(routes)}")

    poi_df = load_poi_data(args.poi_csv)
    print(f"  POI数据: {len(poi_df)}")

    trajectories = load_trajectories(args.trajectories)
    print(f"  轨迹数据: {len(trajectories)}")

    print("\n生成DPO偏好对...")

    print("  1. 路线偏好对...")
    route_pairs = build_route_preference_pairs(routes, poi_df, args.route_pairs)
    print(f"     生成: {len(route_pairs)} 对")

    print("  2. 标题偏好对...")
    title_pairs = build_title_preference_pairs(routes, args.title_pairs)
    print(f"     生成: {len(title_pairs)} 对")

    print("  3. 轨迹偏好对...")
    traj_pairs = build_trajectory_preference_pairs(trajectories, poi_df, args.traj_pairs)
    print(f"     生成: {len(traj_pairs)} 对")

    # 合并所有偏好对
    all_pairs = route_pairs + title_pairs + traj_pairs
    random.shuffle(all_pairs)

    # 保存
    save_dpo_data(all_pairs, args.output)

    print(f"\n✅ DPO偏好数据生成完成!")
    print(f"  总偏好对: {len(all_pairs)}")
    print(f"\n  分布:")
    print(f"    路线偏好: {len(route_pairs)}")
    print(f"    标题偏好: {len(title_pairs)}")
    print(f"    轨迹偏好: {len(traj_pairs)}")


if __name__ == "__main__":
    main()
