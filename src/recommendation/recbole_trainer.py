"""
RecBole 序列推荐训练器
参考: RecBole-master/recbole/quick_start/quick_start.py
"""
import pandas as pd
import os

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

