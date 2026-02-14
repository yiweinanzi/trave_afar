#!/usr/bin/env python3
"""
Qwen3模型下载脚本（Python版本）
使用huggingface_hub下载模型，支持断点续传
"""
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# 设置HF镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODELS_DIR = Path("./models")
LOG_DIR = Path("./logs")

# 模型列表
MODELS = [
    ("Qwen/Qwen3-8B", "Qwen3-8B", "~16GB"),
    ("Qwen/Qwen3-Embedding-4B", "Qwen3-Embedding-4B", "~8GB"),
    ("Qwen/Qwen3-Reranker-4B", "Qwen3-Reranker-4B", "~8GB"),
]


def check_install():
    """检查并安装huggingface_hub"""
    try:
        import huggingface_hub
        print(f"✓ huggingface_hub 已安装 (版本: {huggingface_hub.__version__})")
        return True
    except ImportError:
        print("正在安装 huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub", "-q"])
        print("✓ 安装完成")
        return True


def download_model(repo_id, local_dir, size_info, log_file):
    """下载单个模型"""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 启动下载: {repo_id} ({size_info})")
    print(f"本地目录: {local_dir}")
    print(f"日志文件: {log_file}")

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 打开日志文件
    with open(log_path, "a") as f:
        f.write(f"\n=== 下载开始 {datetime.now()} ===\n")
        f.write(f"模型: {repo_id}\n")
        f.write(f"目录: {local_dir}\n")

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )

        print(f"✓ {repo_id} 下载完成！")
        with open(log_path, "a") as f:
            f.write(f"=== 下载完成 {datetime.now()} ===\n")
        return True

    except Exception as e:
        print(f"✗ {repo_id} 下载失败: {e}")
        with open(log_path, "a") as f:
            f.write(f"=== 下载失败 {datetime.now()}: {e} ===\n")
        return False


def check_status():
    """检查下载状态"""
    print("\n" + "=" * 50)
    print("下载状态检查")
    print("=" * 50)

    for repo_id, local_name, size_info in MODELS:
        local_dir = MODELS_DIR / local_name
        if local_dir.exists():
            # 计算目录大小
            total_size = sum(f.stat().st_size for f in local_dir.rglob('*') if f.is_file())
            size_gb = total_size / (1024 ** 3)
            file_count = len(list(local_dir.rglob('*')))
            print(f"{local_name}: {size_gb:.2f}GB ({file_count} files)")
        else:
            print(f"{local_name}: 尚未开始")


def main():
    print("=" * 50)
    print("Qwen3系列模型下载脚本")
    print("=" * 50)
    print(f"模型目录: {MODELS_DIR.absolute()}")
    print(f"日志目录: {LOG_DIR.absolute()}")
    print(f"HF镜像: {os.environ.get('HF_ENDPOINT', 'default')}")
    print("=" * 50)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 检查参数
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        check_status()
        return

    # 检查依赖
    check_install()

    # 询问是否后台运行
    background = False
    if len(sys.argv) > 1 and sys.argv[1] == "bg":
        background = True

    if background:
        print("\n后台下载模式...")
        # 后台下载每个模型
        for repo_id, local_name, size_info in MODELS:
            log_file = LOG_DIR / f"download_{local_name.replace('-', '_')}.log"
            local_dir = MODELS_DIR / local_name

            # 使用nohup后台运行
            cmd = [
                "nohup", sys.executable, __file__,
                "download-one", repo_id, str(local_dir), str(log_file)
            ]
            subprocess.Popen(cmd, stdout=open("/dev/null", "w"), stderr=open("/dev/null", "w"))
            print(f"已启动后台下载: {local_name}")
            print(f"  日志: tail -f {log_file}")
        print("\n所有下载任务已在后台启动")
        return

    # 前台下载模式
    print("\n前台下载模式（Ctrl+C 可中断，支持断点续传）")
    print("如需后台下载，请使用: python scripts/download_models.py bg")
    print()

    for i, (repo_id, local_name, size_info) in enumerate(MODELS, 1):
        print(f"\n[{i}/{len(MODELS)}] 准备下载: {local_name} ({size_info})")
        local_dir = MODELS_DIR / local_name
        log_file = LOG_DIR / f"download_{local_name.replace('-', '_')}.log"

        success = download_model(repo_id, local_dir, size_info, log_file)

        if not success:
            print(f"警告: {local_name} 下载失败，请检查日志: {log_file}")

    print("\n" + "=" * 50)
    print("下载任务完成！")
    print("=" * 50)
    check_status()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "download-one":
        # 单个模型下载（用于后台调用）
        _, _, repo_id, local_dir, log_file = sys.argv
        download_model(repo_id, Path(local_dir), "", log_file)
    else:
        main()
