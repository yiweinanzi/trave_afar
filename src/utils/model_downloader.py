"""
下载必要的模型到本地
使用 modelscope 镜像加速
"""
import os

MODEL_DIR = "/root/autodl-tmp/goafar_project/models"

def download_bge_m3():
    """下载 BGE-M3 模型"""
    print("="*60)
    print("下载 BGE-M3 模型")
    print("="*60)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    try:
        # 方法1: 使用 modelscope
        print("\n尝试使用 ModelScope 镜像下载...")
        from modelscope import snapshot_download
        
        model_dir = snapshot_download(
            'Xorbits/bge-m3',  # ModelScope上的BGE-M3
            cache_dir=MODEL_DIR
        )
        print(f"✓ 模型下载完成: {model_dir}")
        return model_dir
        
    except Exception as e1:
        print(f"ModelScope 下载失败: {e1}")
        
        try:
            # 方法2: 使用 huggingface 官方源（较慢）
            print("\n尝试使用 HuggingFace 官方源下载...")
            from transformers import AutoModel, AutoTokenizer
            
            model_name = "BAAI/bge-m3"
            AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_DIR)
            AutoModel.from_pretrained(model_name, cache_dir=MODEL_DIR)
            
            print(f"✓ 模型下载完成!")
            return MODEL_DIR
            
        except Exception as e2:
            print(f"HuggingFace 下载失败: {e2}")
            print("\n建议:")
            print("1. 检查网络连接")
            print("2. 手动下载模型到", MODEL_DIR)
            print("3. 或使用已有的模型缓存")
            return None

if __name__ == "__main__":
    download_bge_m3()
    print("\n✓ 模型下载流程完成")

