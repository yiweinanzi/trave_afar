"""
SHP数据解析器
解析shengfen文件夹中的省份shp文件，提取地理边界和POI信息

用途：
1. 解析省份边界（用于地图展示和地理过滤）
2. 从shp属性中提取POI数据（景点名称、位置、类型等）
3. 将新数据整合到poi.csv

依赖:
    pip install geopandas fiona pyproj
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import pandas as pd

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHP_DIR = PROJECT_ROOT / "data" / "external" / "shengfen"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SHPParser:
    """SHP文件解析器"""

    def __init__(self, shp_dir: Path = None):
        """
        初始化解析器

        Args:
            shp_dir: SHP文件目录，默认为data/external/shengfen
        """
        self.shp_dir = Path(shp_dir) if shp_dir else SHP_DIR
        self.gpdf = None

        # 检查依赖
        try:
            import geopandas as gpd
            self.gpd = gpd
        except ImportError:
            logger.warning("geopandas未安装，尝试使用fiona")
            try:
                import fiona
                self.fiona = fiona
            except ImportError:
                logger.error("请安装 geopandas: pip install geopandas")

    def list_shp_files(self) -> List[Path]:
        """列出所有SHP文件"""
        if not self.shp_dir.exists():
            logger.warning(f"SHP目录不存在: {self.shp_dir}")
            return []

        shp_files = list(self.shp_dir.glob("*.shp"))
        logger.info(f"找到 {len(shp_files)} 个SHP文件")
        return shp_files

    def parse_shp_file(self, shp_path: Path) -> Optional[pd.DataFrame]:
        """
        解析单个SHP文件

        Args:
            shp_path: SHP文件路径

        Returns:
            包含几何和属性的DataFrame
        """
        if self.gpd:
            return self._parse_with_geopandas(shp_path)
        elif hasattr(self, 'fiona'):
            return self._parse_with_fiona(shp_path)
        else:
            logger.error("无可用的SHP解析库")
            return None

    def _parse_with_geopandas(self, shp_path: Path) -> Optional[pd.DataFrame]:
        """使用geopandas解析"""
        try:
            gdf = self.gpd.read_file(shp_path)
            logger.info(f"解析成功: {shp_path.name}, 行数: {len(gdf)}, 列数: {len(gdf.columns)}")
            return gdf
        except Exception as e:
            logger.error(f"解析失败 {shp_path.name}: {e}")
            return None

    def _parse_with_fiona(self, shp_path: Path) -> Optional[pd.DataFrame]:
        """使用fiona解析"""
        try:
            with self.fiona.open(shp_path) as src:
                features = list(src)
                data = []
                for feature in features:
                    row = feature['properties']
                    row['geometry'] = feature['geometry']
                    data.append(row)
                df = pd.DataFrame(data)
                logger.info(f"解析成功: {shp_path.name}, 特征数: {len(features)}")
                return df
        except Exception as e:
            logger.error(f"解析失败 {shp_path.name}: {e}")
            return None

    def extract_province_name(self, shp_path: Path) -> str:
        """
        从文件名提取省份名称

        例如: gansu-260212-free.shp.zip -> gansu -> 甘肃
        """
        name = shp_path.stem.lower()
        # 移除后缀
        for suffix in ["-260212-free", "-260213-free", "-free.shp", ".shp"]:
            name = name.replace(suffix, "")

        # 英文转中文映射
        name_map = {
            "anhui": "安徽", "beijing": "北京", "chongqing": "重庆",
            "fujian": "福建", "gansu": "甘肃", "guangdong": "广东",
            "guangxi": "广西", "hainan": "海南", "hebei": "河北",
            "heilongjiang": "黑龙江", "henan": "河南", "hubei": "湖北",
            "hunan": "湖南", "inner-mongolia": "内蒙古", "jiangsu": "江苏",
            "jiangxi": "江西", "jilin": "吉林", "liaoning": "辽宁",
            "macau": "澳门", "ningxia": "宁夏", "qinghai": "青海",
            "shaanxi": "陕西", "shandong": "山东", "shanghai": "上海",
            "shanxi": "山西", "sichuan": "四川", "tianjin": "天津",
            "tibet": "西藏", "xinjiang": "新疆", "yunnan": "云南",
            "zhejiang": "浙江", "hong-kong": "香港"
        }

        return name_map.get(name, name)

    def get_bbox(self, gdf: pd.DataFrame) -> Tuple[float, float, float, float]:
        """
        获取边界框 (min_lon, min_lat, max_lon, max_lat)
        """
        if self.gpd and 'geometry' in gdf.columns:
            bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
            return (bounds[0], bounds[1], bounds[2], bounds[3])
        return (0, 0, 0, 0)

    def extract_cities(self, gdf: pd.DataFrame) -> List[Dict]:
        """
        从SHP数据中提取城市信息

        Args:
            gdf: GeoDataFrame

        Returns:
            城市信息列表
        """
        cities = []

        # 常见的城市名字段
        city_columns = ['name', 'NAME', 'city', 'CITY', '市', '名称']

        for _, row in gdf.iterrows():
            city_info = {}

            # 查找城市名称
            for col in city_columns:
                if col in row and pd.notna(row[col]):
                    city_info['name'] = str(row[col])
                    break

            # 几何中心
            if 'geometry' in row and hasattr(row['geometry'], 'centroid'):
                centroid = row['geometry'].centroid
                city_info['lat'] = centroid.y
                city_info['lon'] = centroid.x

            if city_info:
                cities.append(city_info)

        return cities

    def extract_pois_from_shp(self, gdf: pd.DataFrame, province: str) -> List[Dict]:
        """
        从SHP数据中提取POI信息（模拟）

        注意：真实SHP文件通常不包含景点信息，
        这里主要是演示如何解析属性并转换为POI格式

        实际应用中，应该从其他数据源（如Yelp、高德API等）获取POI数据
        """
        pois = []

        # 检查是否有旅游景点相关的字段
        poi_columns = [
            'name', 'NAME', 'poi_name', 'POI_NAME',
            'scenic', 'SCENIC', 'tourism', 'TOURISM'
        ]

        for _, row in gdf.iterrows():
            poi = {
                'province': province,
                'source': 'shp'
            }

            # 查找名称
            for col in poi_columns:
                if col in row and pd.notna(row[col]):
                    poi['name'] = str(row[col])
                    break

            # 几何位置
            if 'geometry' in row:
                geom = row['geometry']
                if hasattr(geom, 'centroid'):
                    poi['lat'] = geom.centroid.y
                    poi['lon'] = geom.centroid.x

            # 其他属性
            for col in row.index:
                if col != 'geometry' and col not in poi_columns and pd.notna(row[col]):
                    poi[f'attr_{col}'] = str(row[col])

            if 'name' in poi:
                pois.append(poi)

        logger.info(f"从SHP提取到 {len(pois)} 个潜在POI（仅供参考）")
        return pois


def parse_all_shp_files(shp_dir: Path = None) -> Dict[str, pd.DataFrame]:
    """
    解析所有SHP文件

    Args:
        shp_dir: SHP文件目录

    Returns:
        {省份名: GeoDataFrame} 的字典
    """
    parser = SHPParser(shp_dir)
    shp_files = parser.list_shp_files()

    results = {}
    for shp_file in shp_files:
        province = parser.extract_province_name(shp_file)
        gdf = parser.parse_shp_file(shp_file)
        if gdf is not None:
            results[province] = gdf

    return results


def merge_poi_with_existing(new_pois: List[Dict], existing_poi_csv: Path = None) -> pd.DataFrame:
    """
    合并新POI数据到现有CSV

    Args:
        new_pois: 新POI列表
        existing_poi_csv: 现有POI CSV文件路径

    Returns:
        合并后的DataFrame
    """
    if existing_poi_csv is None:
        existing_poi_csv = PROJECT_ROOT / "data" / "poi.csv"

    # 读取现有数据
    if existing_poi_csv.exists():
        existing_df = pd.read_csv(existing_poi_csv)
        logger.info(f"现有POI数量: {len(existing_df)}")
    else:
        existing_df = pd.DataFrame(columns=[
            'poi_id', 'name', 'lat', 'lon', 'open_min', 'close_min',
            'stay_min', 'province', 'city', 'description', 'time_str'
        ])

    # 构造新POI DataFrame
    if new_pois:
        new_df = pd.DataFrame(new_pois)
        # 分配新的poi_id（简单自增）
        max_id = existing_df['poi_id'].astype(str).str.lstrip('0').astype(int).max() if len(existing_df) > 0 else 0
        new_df['poi_id'] = [f"{i:06d}" for i in range(max_id + 1, max_id + 1 + len(new_df))]

        # 补充缺失列
        for col in existing_df.columns:
            if col not in new_df.columns:
                new_df[col] = None

        # 合并
        merged_df = pd.concat([existing_df, new_df], ignore_index=True)
        logger.info(f"合并后POI数量: {len(merged_df)}")

        return merged_df

    return existing_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='SHP数据解析器')
    parser.add_argument('--action', choices=['list', 'parse', 'merge'], default='list',
                        help='操作类型: list-列出文件, parse-解析文件, merge-合并到poi.csv')
    parser.add_argument('--shp-dir', type=str, default=None, help='SHP文件目录')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径')
    parser.add_argument('--test', action='store_true', help='测试模式')

    args = parser.parse_args()

    if args.action == 'list':
        shp_parser = SHPParser(Path(args.shp_dir) if args.shp_dir else None)
        files = shp_parser.list_shp_files()
        for f in files:
            province = shp_parser.extract_province_name(f)
            print(f"  {f.name} -> {province}")

    elif args.action == 'parse':
        results = parse_all_shp_files(Path(args.shp_dir) if args.shp_dir else None)
        print(f"\n解析结果: {len(results)} 个省份")
        for province, gdf in results.items():
            bbox = SHPParser().get_bbox(gdf)
            print(f"  {province}: {len(gdf)} 行, 边界: {bbox}")

    elif args.action == 'merge':
        # 注意：实际使用时应该从真实数据源获取POI
        print("merge功能需要配合真实POI数据源使用")
        print("当前SHP文件主要包含地理边界信息，不包含详细POI数据")
