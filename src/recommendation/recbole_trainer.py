"""
RecBole 序列推荐训练器 & 在线推理 Provider
参考: RecBole-master/recbole/quick_start/quick_start.py
"""
import pandas as pd
import os
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from utils.id_mapping import normalize_poi_id

def export_recbole_data(events_csv='data/user_events.csv',
                       output_dir='outputs/recbole/custom'):
    """
    导出RecBole格式的交互数据
    
    Args:
        events_csv: 用户事件CSV文件
        output_dir: 输出目录
    
    Returns:
        str: 输出文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取用户事件
    df = pd.read_csv(events_csv)
    df['poi_id'] = df['poi_id'].apply(normalize_poi_id)

    poi_csv = 'data/poi.csv'
    if os.path.exists(poi_csv):
        poi_df = pd.read_csv(poi_csv)
        poi_df['poi_id'] = poi_df['poi_id'].apply(normalize_poi_id)
        valid_ids = set(poi_df['poi_id'])
        df = df[df['poi_id'].isin(valid_ids)].copy()
    print(f"加载 {len(df)} 条用户事件")
    print(f"  用户数: {df['user_id'].nunique()}")
    print(f"  POI数: {df['poi_id'].nunique()}")
    print(f"  行为分布: {dict(df['action'].value_counts())}")
    
    # 过滤正反馈
    df = df[df['action'].isin(['click', 'fav', 'visit'])].copy()
    df = df.sort_values(['user_id', 'timestamp'])
    
    print(f"\n过滤后: {len(df)} 条正反馈记录")
    
    # 导出为RecBole格式（tab分隔，无表头）
    output_file = f"{output_dir}/goafar.inter"
    df[['user_id', 'poi_id', 'timestamp']].to_csv(
        output_file,
        sep='\t',
        header=False,
        index=False
    )
    
    print(f"\n✓ 导出 RecBole 数据: {output_file}")
    print(f"  格式: user_id\\tpoi_id\\ttimestamp")
    
    # 显示样例
    print(f"\n样例（前5行）:")
    with open(output_file, 'r') as f:
        for i, line in enumerate(f):
            if i < 5:
                print(f"  {line.strip()}")
    
    return output_file

def train_recbole_model(config_file='configs/recbole.yaml', gpu_id=0):
    """
    训练RecBole模型
    
    Args:
        config_file: 配置文件路径
        gpu_id: GPU ID（-1表示使用CPU）
    
    Returns:
        训练结果
    """
    print("\n" + "="*60)
    print("RecBole 模型训练")
    print("="*60)
    
    try:
        from recbole.quick_start import run_recbole
        
        print(f"配置文件: {config_file}")
        print(f"GPU ID: {gpu_id}")
        
        # 运行RecBole训练
        result = run_recbole(
            model='SASRec',
            dataset='custom',
            config_file_list=[config_file]
        )
        
        print("\n✓ 训练完成")
        return result
        
    except ImportError as e:
        print(f"错误: RecBole未正确安装 - {e}")
        print("\n替代方案:")
        print("  1. 使用流行度召回代替序列推荐")
        print("  2. 或安装RecBole: pip install recbole")
        return None
    except Exception as e:
        print(f"训练失败: {e}")
        return None

if __name__ == "__main__":
    # 导出数据
    export_recbole_data()

    # 训练模型（需要GPU，可选）
    print("\n" + "="*60)
    print("注意：RecBole训练需要GPU和较长时间")
    print("如果跳过训练，系统将使用流行度召回作为替代")
    print("="*60)

    # train_recbole_model()


class RecBoleProvider:
    """
    RecBole 模型在线推理 Provider

    功能：
    - 加载训练好的 RecBole 模型
    - 支持用户个性化推荐预测
    - 模型加载失败时自动降级到流行度召回
    - 支持用户冷启动（无历史行为时回退）
    """

    def __init__(
        self,
        model_path: str = "outputs/recbole/saved",
        config_file: str = "configs/recbole.yaml",
        use_gpu: bool = True,
        fallback_to_popular: bool = True
    ):
        """
        初始化 RecBole Provider

        Args:
            model_path: 训练好的模型保存路径
            config_file: RecBole 配置文件路径
            use_gpu: 是否使用 GPU
            fallback_to_popular: 模型加载失败时是否回退到流行度
        """
        self.model_path = Path(model_path)
        self.config_file = config_file
        self.use_gpu = use_gpu
        self.fallback_to_popular = fallback_to_popular

        self.model = None
        self.dataset = None
        self.user_history = None
        self.item_popularity = None
        self.available = False

        # 尝试加载模型
        self._load_model()

    def _load_model(self):
        """加载 RecBole 模型"""
        try:
            import torch
            from recbole.config import Config
            from recbole.data import create_dataset, data_preparation
            from recbole.utils import get_model, init_seed

            print("\n" + "=" * 60)
            print("加载 RecBole 模型")
            print("=" * 60)

            # 检查模型文件是否存在
            if not self.model_path.exists():
                raise FileNotFoundError(f"模型路径不存在: {self.model_path}")

            # 查找最新的 checkpoint
            checkpoint_files = list(self.model_path.glob("*.pth"))
            if not checkpoint_files:
                raise FileNotFoundError(f"未找到模型文件: {self.model_path}/*.pth")

            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            print(f"找到 checkpoint: {latest_checkpoint}")

            # 加载配置
            config = Config(config_file_list=[self.config_file])
            config['checkpoint_dir'] = str(self.model_path)

            # 创建数据集
            dataset = create_dataset(config)
            self.dataset = dataset

            # 加载模型
            init_seed(config['seed'], config['reproducibility'])
            model = get_model(config['model'])(config, dataset).to(config['device'])

            # 加载权重
            checkpoint = torch.load(latest_checkpoint, map_location=config['device'])
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()

            self.model = model
            self.config = config
            self.available = True

            print(f"✓ 模型加载成功")
            print(f"  模型类型: {config['model']}")
            print(f"  用户数: {dataset.user_num}")
            print(f"  物品数: {dataset.item_num}")

            # 预加载用户历史和物品流行度（用于降级）
            self._preload_fallback_data()

        except ImportError as e:
            print(f"⚠️ RecBole 未安装: {e}")
            if self.fallback_to_popular:
                print("  将使用流行度召回作为替代")
        except FileNotFoundError as e:
            print(f"⚠️ 模型文件不存在: {e}")
            if self.fallback_to_popular:
                print("  将使用流行度召回作为替代")
        except Exception as e:
            print(f"⚠️ 模型加载失败: {e}")
            if self.fallback_to_popular:
                print("  将使用流行度召回作为替代")
            import traceback
            traceback.print_exc()

    def _preload_fallback_data(self):
        """预加载降级所需的数据"""
        try:
            # 加载用户事件数据
            events_csv = "data/user_events.csv"
            if os.path.exists(events_csv):
                events = pd.read_csv(events_csv)
                events['poi_id'] = events['poi_id'].apply(normalize_poi_id)

                # 计算物品流行度
                ACTION_WEIGHT = {"click": 1.0, "fav": 2.0, "visit": 3.0}
                events["weight"] = events["action"].map(ACTION_WEIGHT).fillna(1.0)
                self.item_popularity = events.groupby("poi_id")["weight"].sum().to_dict()

                print(f"  预加载流行度数据: {len(self.item_popularity)} 个物品")
        except Exception as e:
            print(f"  预加载降级数据失败: {e}")

    def predict(
        self,
        user_id: str,
        topk: int = 30,
        poi_df: Optional[pd.DataFrame] = None,
        filter_history: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        预测用户的 Top-K 推荐物品

        Args:
            user_id: 用户 ID
            topk: 返回的推荐物品数量
            poi_df: POI 数据（用于过滤和添加元信息）
            filter_history: 是否过滤用户历史交互过的物品

        Returns:
            (recommendations_df, metadata)
            - recommendations_df: 包含 poi_id 和 score 的 DataFrame
            - metadata: 包含推荐来源、用户信息等元数据
        """
        metadata = {
            "user_id": user_id,
            "method": "none",
            "num_interactions": 0,
            "filtered": False
        }

        # 如果模型不可用，降级到流行度
        if not self.available:
            return self._predict_by_popularity(topk, poi_df, metadata)

        try:
            import torch

            # 将 user_id 转换为内部 ID
            try:
                internal_user_id = self.dataset.token2id(self.dataset.uid_field, str(user_id))
            except KeyError:
                # 用户不在训练集中（冷启动）
                metadata["method"] = "cold_start"
                print(f"  用户 {user_id} 不在训练集中，使用冷启动策略")
                return self._predict_by_popularity(topk, poi_df, metadata)

            # 获取用户历史序列
            user_seq = self.dataset.get_inter_items(
                self.dataset.uid_field,
                internal_user_id
            ).tolist()

            metadata["num_interactions"] = len(user_seq)

            if len(user_seq) == 0:
                # 无历史行为
                metadata["method"] = "no_history"
                print(f"  用户 {user_id} 无历史行为，使用流行度推荐")
                return self._predict_by_popularity(topk, poi_df, metadata)

            # 构造输入序列
            input_seq = torch.tensor([user_seq], dtype=torch.long).to(self.config['device'])

            # 预测
            with torch.no_grad():
                scores = self.model.full_sort_predict(input_seq)
                scores = scores[0]  # 取第一个（唯一的）用户

                # 如果需要过滤历史物品
                if filter_history and len(user_seq) > 0:
                    scores[user_seq] = -float('inf')
                    metadata["filtered"] = True

                # 获取 Top-K
                topk_scores, topk_indices = torch.topk(scores, min(topk * 2, len(scores)))

                # 转换回原始 item_id
                recommended_items = [
                    self.dataset.id2token(self.dataset.iid_field, idx.item())
                    for idx in topk_indices
                ]
                recommended_scores = topk_scores.cpu().numpy().tolist()

            # 构建 DataFrame
            rec_df = pd.DataFrame({
                'poi_id': recommended_items,
                'recbole_score': recommended_scores
            })

            # 归一化分数
            if len(rec_df) > 0 and rec_df['recbole_score'].max() > rec_df['recbole_score'].min():
                rec_df['recbole_score'] = (
                    (rec_df['recbole_score'] - rec_df['recbole_score'].min()) /
                    (rec_df['recbole_score'].max() - rec_df['recbole_score'].min())
                )
            else:
                rec_df['recbole_score'] = 1.0

            # 过滤到 POI 数据中的有效物品
            if poi_df is not None and len(poi_df) > 0:
                valid_ids = set(poi_df['poi_id'])
                rec_df = rec_df[rec_df['poi_id'].isin(valid_ids)].copy()

            # 取前 topk
            rec_df = rec_df.head(topk).reset_index(drop=True)

            metadata["method"] = "recbole"
            return rec_df, metadata

        except Exception as e:
            print(f"⚠️ RecBole 预测失败: {e}，降级到流行度")
            import traceback
            traceback.print_exc()
            return self._predict_by_popularity(topk, poi_df, metadata)

    def _predict_by_popularity(
        self,
        topk: int,
        poi_df: Optional[pd.DataFrame],
        metadata: Dict
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        降级策略：基于流行度的推荐

        Args:
            topk: 推荐数量
            poi_df: POI 数据
            metadata: 元数据（会被更新）

        Returns:
            (recommendations_df, metadata)
        """
        metadata["method"] = "popularity"

        if self.item_popularity is None:
            # 如果没有预加载数据，返回空结果
            print("  ⚠️ 流行度数据不可用")
            return pd.DataFrame(columns=['poi_id', 'recbole_score']), metadata

        # 按 popularity 排序
        sorted_items = sorted(
            self.item_popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 构建结果
        rec_df = pd.DataFrame({
            'poi_id': [item_id for item_id, _ in sorted_items],
            'recbole_score': [score for _, score in sorted_items]
        })

        # 归一化
        if len(rec_df) > 0:
            max_score = max(rec_df['recbole_score'].max(), 1e-8)
            rec_df['recbole_score'] = rec_df['recbole_score'] / max_score

        # 过滤到 POI 数据中的有效物品
        if poi_df is not None and len(poi_df) > 0:
            valid_ids = set(poi_df['poi_id'])
            rec_df = rec_df[rec_df['poi_id'].isin(valid_ids)].copy()

        # 取前 topk
        rec_df = rec_df.head(topk).reset_index(drop=True)

        return rec_df, metadata

    def get_user_history_length(self, user_id: str) -> int:
        """
        获取用户历史交互数量

        Args:
            user_id: 用户 ID

        Returns:
            历史交互数量
        """
        if not self.available:
            return 0

        try:
            internal_user_id = self.dataset.token2id(self.dataset.uid_field, str(user_id))
            user_seq = self.dataset.get_inter_items(
                self.dataset.uid_field,
                internal_user_id
            ).tolist()
            return len(user_seq)
        except:
            return 0
