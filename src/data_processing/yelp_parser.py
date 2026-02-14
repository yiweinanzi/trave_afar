"""
Yelp数据集解析器
解析Yelp开放数据集，补充POI属性和评价信号

数据格式：
Yelp-JSON.zip 包含多个JSON文件：
- yelp_academic_dataset_business.json: 商家信息
- yelp_academic_dataset_review.json: 用户评论
- yelp_academic_dataset_user.json: 用户信息
- yelp_academic_dataset_checkin.json: 签到记录
- yelp_academic_dataset_tip.json: 用户提示

用途：
1. 补充POI属性（类别、评分、营业时间）
2. 情感分析
3. 用户偏好信号
"""
import json
import gzip
from pathlib import Path
from typing import Dict, List, Optional
import logging

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
YELP_JSON = PROJECT_ROOT / "data" / "external" / "Yelp-JSON.zip"
YELP_DIR = PROJECT_ROOT / "data" / "external" / "Yelp-JSON"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class YelpParser:
    """Yelp数据集解析器"""

    def __init__(self, yelp_dir: Path = None):
        """
        初始化解析器

        Args:
            yelp_dir: Yelp数据目录
        """
        self.yelp_dir = Path(yelp_dir) if yelp_dir else YELP_DIR
        self.data_files = {}

        # 检查目录
        if not self.yelp_dir.exists():
            logger.warning(f"Yelp目录不存在: {self.yelp_dir}")
            logger.info("尝试使用原始ZIP文件...")
            self._extract_from_zip()

        # 扫描数据文件
        self._scan_files()

    def _extract_from_zip(self):
        """从ZIP中提取部分数据（示例）"""
        import zipfile

        if not YELP_JSON.exists():
            logger.error(f"Yelp ZIP文件不存在: {YELP_JSON}")
            return

        # 只提取business JSON（用于POI属性）
        logger.info("从ZIP中提取business数据...")
        self.yelp_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(YELP_JSON, 'r') as zip_ref:
                # 列出文件
                files = [f for f in zip_ref.namelist() if 'business' in f]
                logger.info(f"找到 {len(files)} 个business相关文件")

                # 提取第一个business文件
                for f in files:
                    if f.endswith('.json'):
                        zip_ref.extract(f, self.yelp_dir.parent)
                        logger.info(f"提取: {f}")
                        break

        except Exception as e:
            logger.error(f"解压失败: {e}")

    def _scan_files(self):
        """扫描JSON数据文件"""
        if not self.yelp_dir.exists():
            return

        patterns = {
            'business': 'yelp_academic_dataset_business.json',
            'review': 'yelp_academic_dataset_review.json',
            'user': 'yelp_academic_dataset_user.json',
            'checkin': 'yelp_academic_dataset_checkin.json',
            'tip': 'yelp_academic_dataset_tip.json'
        }

        for key, pattern in patterns.items():
            # 查找匹配文件
            matches = list(self.yelp_dir.rglob(pattern)) if self.yelp_dir.exists() else []
            if matches:
                self.data_files[key] = matches[0]
                logger.info(f"找到 {key}: {matches[0].name}")

    def parse_business_json(self, max_lines: int = None) -> pd.DataFrame:
        """
        解析商家信息JSON

        Args:
            max_lines: 最大读取行数

        Returns:
            商家DataFrame
        """
        business_file = self.data_files.get('business')
        if not business_file:
            logger.error("business文件不存在")
            return pd.DataFrame()

        logger.info(f"正在解析business数据...")
        records = []

        try:
            with open(business_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_lines and i >= max_lines:
                        break

                    try:
                        data = json.loads(line)

                        # 只保留旅游景点相关的
                        categories = data.get('categories', [])
                        tourism_keywords = ['museum', 'park', 'landmark', 'historic',
                                         'tourist', 'attraction', 'monument', 'temple',
                                         'church', 'castle', 'gallery', 'garden']

                        if not any(kw in ' '.join(categories).lower() for kw in tourism_keywords):
                            continue

                        records.append({
                            'business_id': data.get('business_id'),
                            'name': data.get('name'),
                            'address': data.get('address'),
                            'city': data.get('city'),
                            'state': data.get('state'),
                            'postal_code': data.get('postal_code'),
                            'latitude': data.get('latitude'),
                            'longitude': data.get('longitude'),
                            'stars': data.get('stars'),
                            'review_count': data.get('review_count'),
                            'is_open': data.get('is_open'),
                            'categories': ','.join(categories) if categories else '',
                            'attributes': json.dumps(data.get('attributes', {})),
                            'hours': json.dumps(data.get('hours', {}))
                        })

                    except json.JSONDecodeError:
                        continue

                    if (i + 1) % 10000 == 0:
                        logger.info(f"已处理 {i + 1} 行，找到 {len(records)} 个旅游相关商家")

        except Exception as e:
            logger.error(f"解析失败: {e}")

        df = pd.DataFrame(records)
        logger.info(f"解析完成: {len(df)} 个旅游相关商家")

        return df

    def parse_reviews_json(self, business_ids: List[str] = None,
                           max_lines: int = None) -> pd.DataFrame:
        """
        解析评论JSON

        Args:
            business_ids: 只解析指定商家的评论
            max_lines: 最大读取行数

        Returns:
            评论DataFrame
        """
        review_file = self.data_files.get('review')
        if not review_file:
            logger.warning("review文件不存在")
            return pd.DataFrame()

        logger.info(f"正在解析review数据...")
        records = []

        business_set = set(business_ids) if business_ids else None

        try:
            with open(review_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_lines and i >= max_lines:
                        break

                    try:
                        data = json.loads(line)

                        # 过滤商家
                        if business_set and data.get('business_id') not in business_set:
                            continue

                        records.append({
                            'review_id': data.get('review_id'),
                            'user_id': data.get('user_id'),
                            'business_id': data.get('business_id'),
                            'stars': data.get('stars'),
                            'useful': data.get('useful'),
                            'funny': data.get('funny'),
                            'cool': data.get('cool'),
                            'text': data.get('text', '')[:500],  # 限制长度
                            'date': data.get('date')
                        })

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error(f"解析失败: {e}")

        df = pd.DataFrame(records)
        logger.info(f"解析完成: {len(df)} 条评论")

        return df

    def extract_sentiment_from_reviews(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """
        从评论中提取情感倾向

        Args:
            reviews_df: 评论DataFrame

        Returns:
            商家情感统计DataFrame
        """
        if reviews_df.empty:
            return pd.DataFrame()

        # 简单情感分析（基于星级）
        sentiment_stats = reviews_df.groupby('business_id').agg({
            'stars': ['mean', 'count'],
            'useful': 'mean',
            'text': lambda x: ' '.join(x.tolist()[:10])  # 取前10条文本
        }).reset_index()

        sentiment_stats.columns = ['business_id', 'avg_stars', 'review_count',
                                   'avg_useful', 'sample_texts']

        return sentiment_stats

    def export_to_csv(self, output_dir: Path = None, max_businesses: int = 50000):
        """
        导出数据到CSV

        Args:
            output_dir: 输出目录
            max_businesses: 最大商家数
        """
        if output_dir is None:
            output_dir = PROJECT_ROOT / "data" / "yelp"

        output_dir.mkdir(parents=True, exist_ok=True)

        # 解析商家信息
        business_df = self.parse_business_json(max_lines=max_businesses)

        if not business_df.empty:
            business_file = output_dir / "yelp_businesses.csv"
            business_df.to_csv(business_file, index=False, encoding='utf-8')
            logger.info(f"导出商家信息: {business_file}")

            # 解析评论（仅对评分高的商家）
            top_businesses = business_df.nlargest(1000, 'review_count')['business_id'].tolist()
            reviews_df = self.parse_reviews_json(business_ids=top_businesses, max_lines=100000)

            if not reviews_df.empty:
                reviews_file = output_dir / "yelp_reviews.csv"
                reviews_df.to_csv(reviews_file, index=False, encoding='utf-8')
                logger.info(f"导出评论: {reviews_file}")

                # 情感分析
                sentiment_df = self.extract_sentiment_from_reviews(reviews_df)
                sentiment_file = output_dir / "yelp_sentiment.csv"
                sentiment_df.to_csv(sentiment_file, index=False)
                logger.info(f"导出情感分析: {sentiment_file}")


def match_yelp_to_poi(yelp_df: pd.DataFrame, poi_df: pd.DataFrame) -> pd.DataFrame:
    """
    将Yelp商家匹配到现有POI

    匹配策略：
    1. 名称相似度
    2. 地理距离（<500米）

    Args:
        yelp_df: Yelp商家DataFrame
        poi_df: 现有POI DataFrame

    Returns:
        匹配结果DataFrame
    """
    from difflib import SequenceMatcher

    matches = []

    for _, yelp_row in yelp_df.iterrows():
        yelp_name = yelp_row['name'].lower()
        yelp_lat = yelp_row['latitude']
        yelp_lon = yelp_row['longitude']

        best_match = None
        best_score = 0

        for _, poi_row in poi_df.iterrows():
            poi_name = poi_row['name'].lower()

            # 名称相似度
            name_ratio = SequenceMatcher(None, yelp_name, poi_name).ratio()

            # 地理距离
            from math import radians, cos, sin, asin, sqrt
            def haversine(lat1, lon1, lat2, lon2):
                lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                return 6371000 * 2 * asin(sqrt(a))

            if pd.notna(poi_row['lat']) and pd.notna(poi_row['lon']):
                distance = haversine(yelp_lat, yelp_lon, poi_row['lat'], poi_row['lon'])
            else:
                distance = float('inf')

            # 综合评分
            if distance < 500 and name_ratio > 0.5:
                score = name_ratio * (1 - distance / 500)
                if score > best_score:
                    best_score = score
                    best_match = poi_row['poi_id']

        if best_match:
            matches.append({
                'poi_id': best_match,
                'yelp_id': yelp_row['business_id'],
                'name_score': best_score,
                'yelp_stars': yelp_row['stars'],
                'yelp_review_count': yelp_row['review_count']
            })

    return pd.DataFrame(matches)


def analyze_yelp_data(max_lines: int = 10000):
    """分析Yelp数据集"""
    parser = YelpParser()

    print("\n正在解析Yelp商家数据...")
    business_df = parser.parse_business_json(max_lines=max_lines)

    if business_df.empty:
        print("没有商家数据")
        return

    # 统计信息
    stats = {
        'total_businesses': len(business_df),
        'unique_cities': business_df['city'].nunique(),
        'unique_states': business_df['state'].nunique(),
        'avg_stars': business_df['stars'].mean(),
        'avg_reviews': business_df['review_count'].mean()
    }

    print(f"\nYelp数据分析:")
    print(f"  总商家数: {stats['total_businesses']:,}")
    print(f"  覆盖城市数: {stats['unique_cities']}")
    print(f"  覆盖州数: {stats['unique_states']}")
    print(f"  平均评分: {stats['avg_stars']:.2f}")
    print(f"  平均评论数: {stats['avg_reviews']:.1f}")

    # 类别分布
    all_categories = []
    for cats in business_df['categories'].dropna():
        all_categories.extend(cats.split(','))

    from collections import Counter
    cat_counts = Counter(all_categories)
    print(f"\n热门类别:")
    for cat, count in cat_counts.most_common(10):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Yelp数据集解析器')
    parser.add_argument('--action', choices=['analyze', 'export', 'match'],
                        default='analyze', help='操作类型')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录')
    parser.add_argument('--max-lines', type=int, default=50000,
                        help='最大读取行数')

    args = parser.parse_args()

    if args.action == 'analyze':
        analyze_yelp_data(max_lines=args.max_lines)

    elif args.action == 'export':
        yelp = YelpParser()
        output_dir = Path(args.output_dir) if args.output_dir else None
        yelp.export_to_csv(output_dir, max_businesses=args.max_lines)

    elif args.action == 'match':
        # 匹配Yelp数据到现有POI
        yelp = YelpParser()
        yelp_df = yelp.parse_business_json(max_lines=args.max_lines)

        poi_df = pd.read_csv(PROJECT_ROOT / "data" / "poi.csv")
        matches_df = match_yelp_to_poi(yelp_df, poi_df)

        output = Path(args.output_dir) / "yelp_poi_matches.csv" if args.output_dir else PROJECT_ROOT / "data" / "yelp_poi_matches.csv"
        matches_df.to_csv(output, index=False)
        print(f"匹配结果: {len(matches_df)} 对，保存到 {output}")
