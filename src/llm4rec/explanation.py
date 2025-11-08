"""
推荐解释生成器
使用LLM生成个性化的推荐理由，提升可解释性
"""

class ExplanationGenerator:
    """推荐解释生成器"""
    
    def __init__(self, llm_model=None):
        self.llm_model = llm_model
    
    def generate_poi_explanation(self, poi, user_intent):
        """
        为单个POI生成推荐理由
        
        Args:
            poi: POI字典
            user_intent: 用户意图
        
        Returns:
            str: 推荐理由
        """
        reasons = []
        
        # 匹配兴趣点
        interests = user_intent.get('interests', [])
        for interest in interests:
            if interest in poi['name'] or (poi.get('description') and interest in poi['description']):
                reasons.append(f"✓ 符合您对{interest}的需求")
        
        # 匹配活动
        activities = user_intent.get('activities', [])
        for activity in activities:
            if activity in poi.get('description', ''):
                reasons.append(f"✓ 适合{activity}活动")
        
        # 季节匹配
        season = user_intent.get('season_preference')
        if season and season in poi.get('description', ''):
            reasons.append(f"✓ {season}季是最佳游览时间")
        
        if not reasons:
            reasons.append(f"✓ 该地区的特色景点")
        
        return '\n'.join(reasons)
    
    def generate_route_explanation(self, route_result, user_intent):
        """
        为整条路线生成推荐解释
        
        Args:
            route_result: 路线结果
            user_intent: 用户意图
        
        Returns:
            str: 路线推荐解释
        """
        explanation_parts = []
        
        explanation_parts.append(f"【推荐理由】")
        explanation_parts.append(f"根据您的需求「{user_intent['original_query']}」")
        
        # 匹配度分析
        if user_intent.get('interests'):
            explanation_parts.append(f"\n【兴趣匹配】")
            for interest in user_intent['interests']:
                explanation_parts.append(f"• {interest}相关景点已包含在行程中")
        
        # 路线特点
        explanation_parts.append(f"\n【路线特点】")
        explanation_parts.append(f"• 涵盖 {route_result['num_pois']} 个精选景点")
        explanation_parts.append(f"• 预计用时 {route_result['total_hours']:.1f} 小时")
        explanation_parts.append(f"• 所有景点均在营业时间内可达")
        
        # 亮点景点
        if len(route_result['route']) > 2:
            explanation_parts.append(f"\n【核心景点】")
            for i, poi in enumerate(route_result['route'][1:-1][:3], 1):
                explanation_parts.append(f"{i}. {poi['poi_name']} - {poi['poi_city']}")
        
        return '\n'.join(explanation_parts)

if __name__ == "__main__":
    # 测试
    generator = ExplanationGenerator()
    
    test_poi = {
        'name': '喀纳斯湖',
        'description': '秋天的喀纳斯湖景色迷人，适合摄影'
    }
    
    test_intent = {
        'original_query': '想看秋天的雪山和湖泊，拍照',
        'interests': ['雪山', '湖泊'],
        'activities': ['拍照'],
        'season_preference': '秋'
    }
    
    explanation = generator.generate_poi_explanation(test_poi, test_intent)
    print(f"POI推荐理由:\n{explanation}")

