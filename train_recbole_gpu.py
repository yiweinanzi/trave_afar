"""
RecBole GPU训练脚本
使用GPU训练SASRec序列推荐模型
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def train_recbole_with_gpu(config_file='configs/recbole.yaml', gpu_id=0):
    """
    使用GPU训练RecBole模型
    
    Args:
        config_file: 配置文件路径
        gpu_id: GPU ID
    
    Returns:
        训练结果
    """
    print("="*80)
    print("RecBole 序列推荐模型训练 - GPU加速")
    print("="*80)
    
    # 检查GPU
    import torch
    if not torch.cuda.is_available():
        print("⚠️ 未检测到GPU，将使用CPU训练（较慢）")
        gpu_id = -1
    else:
        gpu_name = torch.cuda.get_device_name(gpu_id)
        gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
        print(f"✓ 使用GPU: {gpu_name}")
        print(f"  显存: {gpu_memory:.1f} GB")
    
    # 准备数据
    from recommendation.recbole_trainer import export_recbole_data
    
    if not os.path.exists('outputs/recbole/custom/goafar.inter'):
        print(f"\n准备RecBole数据...")
        export_recbole_data()
    else:
        print(f"\n✓ RecBole数据已存在")
    
    # 训练模型
    print(f"\n开始训练 SASRec 模型...")
    print(f"  配置文件: {config_file}")
    print(f"  GPU ID: {gpu_id}")
    
    try:
        from recbole.quick_start import run_recbole
        
        result = run_recbole(
            model='SASRec',
            dataset='custom',
            config_file_list=[config_file],
            config_dict={'gpu_id': gpu_id}
        )
        
        print("\n✓ 训练完成！")
        print(f"\n评测结果:")
        print(f"  Best valid score: {result.get('best_valid_score', 'N/A')}")
        print(f"  Test result: {result.get('test_result', 'N/A')}")
        
        return result
        
    except ImportError as e:
        print(f"\n✗ RecBole未正确安装: {e}")
        print("\n解决方案:")
        print("  pip install recbole")
        return None
        
    except Exception as e:
        print(f"\n✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RecBole GPU训练')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--config', type=str, default='configs/recbole.yaml', help='配置文件路径')
    args = parser.parse_args()
    
    result = train_recbole_with_gpu(
        config_file=args.config,
        gpu_id=args.gpu
    )
    
    if result:
        print("\n✓ 模型训练成功！")
        print("\n模型保存位置: outputs/recbole/saved/")
        print("\n下一步: 使用训练好的模型进行推荐")
    else:
        print("\n⚠️ 训练失败或跳过")
        print("系统将使用流行度召回作为替代方案")

