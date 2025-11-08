"""
缓存管理器
提供向量检索、路线规划等的缓存功能
"""
import os
import json
import hashlib
import pickle
import numpy as np
from datetime import datetime, timedelta

class CacheManager:
    """缓存管理器"""
    
    def __init__(self, cache_dir='outputs/cache'):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_ttl = timedelta(hours=24)  # 缓存有效期24小时
    
    def _get_cache_key(self, prefix, params):
        """
        生成缓存键
        
        Args:
            prefix: 缓存前缀
            params: 参数字典
        
        Returns:
            str: 缓存文件路径
        """
        # 将参数序列化为字符串
        param_str = json.dumps(params, sort_keys=True)
        # 生成hash
        param_hash = hashlib.md5(param_str.encode()).hexdigest()
        cache_file = f"{self.cache_dir}/{prefix}_{param_hash}.pkl"
        return cache_file
    
    def get(self, prefix, params):
        """
        获取缓存
        
        Args:
            prefix: 缓存前缀
            params: 参数字典
        
        Returns:
            缓存的数据，如果不存在或过期则返回None
        """
        cache_file = self._get_cache_key(prefix, params)
        
        if not os.path.exists(cache_file):
            return None
        
        # 检查过期
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - file_time > self.cache_ttl:
            print(f"缓存已过期: {cache_file}")
            os.remove(cache_file)
            return None
        
        # 加载缓存
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"✓ 缓存命中: {prefix}")
            return data
        except Exception as e:
            print(f"缓存加载失败: {e}")
            return None
    
    def set(self, prefix, params, data):
        """
        设置缓存
        
        Args:
            prefix: 缓存前缀
            params: 参数字典
            data: 要缓存的数据
        """
        cache_file = self._get_cache_key(prefix, params)
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"✓ 缓存已保存: {prefix}")
        except Exception as e:
            print(f"缓存保存失败: {e}")
    
    def clear(self, prefix=None):
        """
        清除缓存
        
        Args:
            prefix: 如果指定，只清除该前缀的缓存；否则清除所有
        """
        if prefix:
            pattern = f"{prefix}_*.pkl"
        else:
            pattern = "*.pkl"
        
        import glob
        cache_files = glob.glob(f"{self.cache_dir}/{pattern}")
        
        for f in cache_files:
            os.remove(f)
        
        print(f"✓ 清除了 {len(cache_files)} 个缓存文件")

# 全局缓存实例
_cache_manager = None

def get_cache_manager():
    """获取全局缓存管理器实例"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager

if __name__ == "__main__":
    # 测试缓存
    cache = CacheManager()
    
    # 测试设置缓存
    test_params = {'query': '测试查询', 'topk': 10}
    test_data = {'results': [1, 2, 3, 4, 5]}
    
    cache.set('search', test_params, test_data)
    
    # 测试获取缓存
    cached_data = cache.get('search', test_params)
    print(f"缓存数据: {cached_data}")
    
    # 清除缓存
    cache.clear('search')
    print("✓ 缓存测试完成")

