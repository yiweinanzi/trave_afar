"""
构造偏好数据
从历史数据或人工标注生成(prompt, chosen, rejected)对
"""
import os
import pandas as pd
import random

def make_prefs_from_history(output_file='outputs/dpo/prefs.csv'):
    """
    从历史数据构造偏好对
    
    示例：基于用户行为（点赞/收藏/完读）提取偏好
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 示例偏好数据
    # 实际应用中可以从以下来源获取：
    # 1. 用户行为数据（点赞/收藏/完读率高的作为chosen）
    # 2. A/B测试结果（CTR高的作为chosen）
    # 3. 人工标注（标注员对比两个标题，选择更好的）
    
    sample_prefs = [
        {
            'prompt': '给"古城+夜景+步行少"行程写标题',
            'chosen': '西安古城轻走｜夜景串游 4h 不卡点',
            'rejected': '某地城市旅游路线推荐 标题一'
        },
        {
            'prompt': '给"湖泊+拍照+轻松"行程写标题',
            'chosen': '天山南北｜喀纳斯湖-赛里木湖，镜面天空',
            'rejected': '新疆旅游路线推荐'
        },
        {
            'prompt': '给"雪山+徒步+深度游"行程写标题',
            'chosen': '雪域高原｜珠峰大本营-纳木错，触摸世界之巅',
            'rejected': '西藏旅游路线'
        },
        {
            'prompt': '给"草原+骑行+亲子游"行程写标题',
            'chosen': '草原天路｜呼伦贝尔-阿尔山，策马奔腾',
            'rejected': '内蒙古旅游路线推荐'
        },
        {
            'prompt': '给"沙漠+摄影+轻松"行程写标题',
            'chosen': '大漠风光｜敦煌-月牙泉，丝路驼铃',
            'rejected': '甘肃旅游路线'
        },
        {
            'prompt': '给"湖泊+草原+深度游"行程写标题',
            'chosen': '高原明珠｜青海湖-茶卡盐湖，天空之境',
            'rejected': '青海旅游路线推荐'
        },
        {
            'prompt': '给"古城+文化+休闲"行程写标题',
            'chosen': '彩云之南｜大理-丽江，风花雪月',
            'rejected': '云南旅游路线'
        },
        {
            'prompt': '给"峡谷+徒步+挑战"行程写标题',
            'chosen': '川西秘境｜九寨沟-稻城亚丁，人间仙境',
            'rejected': '四川旅游路线推荐'
        }
    ]
    
    df = pd.DataFrame(sample_prefs)
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"✓ 创建偏好数据: {len(df)} 条")
    print(f"保存位置: {output_file}")
    
    return df

if __name__ == "__main__":
    make_prefs_from_history()

