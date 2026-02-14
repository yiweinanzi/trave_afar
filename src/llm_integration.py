"""
大模型集成模块
提供统一的LLM接口，支持Qwen、Llava、API等多种模式
"""
import os
import sys

# 添加src到路径
sys.path.insert(0, os.path.dirname(__file__))

from content_generation.llm_generator import LLMGenerator

class GoAfarLLM:
    """
    GoAfar LLM统一接口
    支持多种大模型和生成模式
    """
    
    def __init__(self, mode='template', model_name='qwen', api_key=None):
        """
        初始化LLM接口
        
        Args:
            mode: 'template' (模板), 'local' (本地模型), 'api' (API调用)
            model_name: 模型名称 ('qwen', 'llava', 'gpt-4')
            api_key: API密钥（API模式需要）
        """
        self.mode = mode
        self.model_name = model_name
        self.api_key = api_key
        
        print(f"初始化 GoAfar LLM 接口:")
        print(f"  模式: {mode}")
        print(f"  模型: {model_name}")
        
        if mode == 'local':
            self.generator = LLMGenerator(
                model_type=model_name,
                use_api=False
            )
        elif mode == 'api':
            self.generator = LLMGenerator(
                model_type=model_name,
                use_api=True
            )
        else:
            # 模板模式
            self.generator = None
            print("  使用模板生成模式（快速且稳定）")
    
    def understand_query(self, query):
        """
        理解用户查询，提取关键信息
        
        Args:
            query: 用户查询文本
        
        Returns:
            dict: {
                'province': 目标省份,
                'interests': 兴趣点列表,
                'duration': 期望天数,
                'style': 旅行风格
            }
        """
        if self.generator:
            return self.generator.analyze_user_intent(query)
        else:
            # 简单的关键词匹配
            return self._simple_intent_analysis(query)
    
    def generate_route_title(self, route_pois, province, query=None):
        """
        生成路线标题
        
        Args:
            route_pois: 路线POI列表
            province: 省份
            query: 用户查询
        
        Returns:
            str: 生成的标题
        """
        if self.generator:
            return self.generator.generate_title(route_pois, province, query)
        else:
            from content_generation.title_generator import generate_title
            return generate_title(route_pois, province, query)
    
    def generate_route_description(self, route_pois, province, total_hours, query=None):
        """
        生成路线描述
        
        Args:
            route_pois: 路线POI列表
            province: 省份
            total_hours: 总时长
            query: 用户查询
        
        Returns:
            str: 生成的描述
        """
        if self.generator:
            return self.generator.generate_description(route_pois, province, total_hours, query)
        else:
            from content_generation.title_generator import generate_description
            return generate_description(route_pois, province, total_hours, query)
    
    def explain_route(self, route_result):
        """
        解释路线推荐结果（可解释性）
        
        Args:
            route_result: 路线结果字典
        
        Returns:
            str: 推荐解释
        """
        explanation_parts = []
        
        explanation_parts.append(f"【推荐理由】")
        explanation_parts.append(f"根据您的需求「{route_result.get('query', '未指定')}」，")
        explanation_parts.append(f"我们为您规划了这条{route_result['province']}精品路线。\n")
        
        explanation_parts.append(f"【路线特点】")
        explanation_parts.append(f"• 涵盖 {route_result['num_pois']} 个精选景点")
        explanation_parts.append(f"• 预计用时 {route_result['total_hours']:.1f} 小时")
        explanation_parts.append(f"• 所有景点均在营业时间内可达\n")
        
        if len(route_result['route']) > 2:
            explanation_parts.append(f"【行程亮点】")
            for i, poi in enumerate(route_result['route'][1:-1][:3], 1):
                explanation_parts.append(f"{i}. {poi['poi_name']} - 停留{poi['stay_min']}分钟")
        
        return '\n'.join(explanation_parts)
    
    def _simple_intent_analysis(self, query):
        """简单的意图分析"""
        provinces = ['新疆', '西藏', '云南', '四川', '甘肃', '青海', '宁夏', '内蒙古']
        
        result = {
            'province': None,
            'interests': [],
            'duration': None,
            'style': '观光游'
        }
        
        for prov in provinces:
            if prov in query:
                result['province'] = prov
                break
        
        # 提取天数
        import re
        days_match = re.search(r'(\d+)[天日]', query)
        if days_match:
            result['duration'] = int(days_match.group(1))
        
        return result

if __name__ == "__main__":
    # 测试不同模式
    print("="*60)
    print("测试模板模式")
    print("="*60)
    
    llm = GoAfarLLM(mode='template')
    
    query = "想去新疆看3天雪山和草原，拍照"
    intent = llm.understand_query(query)
    print(f"\n意图分析: {intent}")
    
    test_route = {
        'query': query,
        'province': '新疆',
        'num_pois': 5,
        'total_hours': 10.5,
        'route': [
            {'poi_name': '乌鲁木齐机场', 'stay_min': 60},
            {'poi_name': '天山天池', 'stay_min': 180},
            {'poi_name': '喀纳斯湖', 'stay_min': 240},
            {'poi_name': '禾木村', 'stay_min': 180},
            {'poi_name': '乌鲁木齐机场', 'stay_min': 0}
        ]
    }
    
    explanation = llm.explain_route(test_route)
    print(f"\n推荐解释:\n{explanation}")

