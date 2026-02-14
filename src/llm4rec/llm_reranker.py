"""
LLM Reranker - 候选重排序
使用LLM对召回的候选POI进行重排序，考虑用户意图和POI间的协同性
"""
import pandas as pd
import json

class LLMReranker:
    """LLM重排序器"""
    
    def __init__(self, llm_model=None, use_template=True):
        """
        初始化
        
        Args:
            llm_model: LLM模型实例
            use_template: 是否使用规则模板
        """
        self.llm_model = llm_model
        self.use_template = use_template
        
        if use_template:
            print("LLM Reranker: 使用规则模板模式")
        else:
            print("LLM Reranker: 使用LLM推理模式")
    
    def rerank(self, candidates_df, user_intent, topk=20):
        """
        对候选POI重排序
        
        Args:
            candidates_df: 候选POI DataFrame
            user_intent: 用户意图字典（由IntentUnderstandingModule生成）
            topk: 返回Top-K
        
        Returns:
            DataFrame: 重排序后的候选
        """
        print(f"\n=== LLM Reranking ===")
        print(f"候选数: {len(candidates_df)}")
        print(f"用户意图: {user_intent.get('interests', [])} + {user_intent.get('activities', [])}")
        
        if self.use_template or self.llm_model is None:
            return self._rule_based_rerank(candidates_df, user_intent, topk)
        else:
            return self._llm_based_rerank(candidates_df, user_intent, topk)
    
    def _rule_based_rerank(self, candidates_df, user_intent, topk):
        """
        基于规则的重排序
        
        考虑因素:
        1. 兴趣匹配度
        2. 地理聚合性（同区域景点加权）
        3. 季节适配度
        4. 活动匹配度
        """
        candidates = candidates_df.copy()
        
        # 初始分数（来自语义检索 semantic_score）
        if 'semantic_score' in candidates.columns:
            candidates['rerank_score'] = candidates['semantic_score'].copy()
        elif 'final_score' in candidates.columns:
            candidates['rerank_score'] = candidates['final_score'].copy()
        else:
            # 如果没有分数列，使用默认值
            candidates['rerank_score'] = 0.5
        
        # 1. 兴趣匹配加权
        interests = user_intent.get('interests', [])
        if interests:
            for interest in interests:
                # 在名称或描述中匹配
                mask = candidates['name'].str.contains(interest, case=False, na=False) | \
                       candidates['description'].str.contains(interest, case=False, na=False)
                candidates.loc[mask, 'rerank_score'] *= 1.2
        
        # 2. 活动匹配加权
        activities = user_intent.get('activities', [])
        if activities:
            for activity in activities:
                mask = candidates['name'].str.contains(activity, case=False, na=False) | \
                       candidates['description'].str.contains(activity, case=False, na=False)
                candidates.loc[mask, 'rerank_score'] *= 1.15
        
        # 3. 地理聚合加权（同城市的POI相互加权）
        if 'city' in candidates.columns:
            city_counts = candidates['city'].value_counts()
            top_cities = city_counts.head(3).index.tolist()
            
            for city in top_cities:
                mask = candidates['city'] == city
                candidates.loc[mask, 'rerank_score'] *= 1.1
        
        # 4. 季节加权（如果有季节信息）
        season = user_intent.get('season_preference')
        if season:
            season_keywords = {
                '春': ['春', '花', '杏'],
                '夏': ['夏', '草原', '湖'],
                '秋': ['秋', '胡杨', '红叶'],
                '冬': ['冬', '雪', '冰']
            }
            keywords = season_keywords.get(season, [])
            for kw in keywords:
                mask = candidates['name'].str.contains(kw, case=False, na=False) | \
                       candidates['description'].str.contains(kw, case=False, na=False)
                candidates.loc[mask, 'rerank_score'] *= 1.1
        
        # 排序并返回Top-K
        candidates = candidates.sort_values('rerank_score', ascending=False).head(topk)
        
        print(f"✓ 重排序完成，返回 Top {len(candidates)}")
        return candidates
    
    def _llm_based_rerank(self, candidates_df, user_intent, topk):
        """
        基于LLM的重排序
        
        使用LLM理解候选POI与用户意图的匹配度
        """
        # 构建prompt
        candidates_info = []
        for idx, row in candidates_df.head(50).iterrows():  # 限制LLM处理数量
            candidates_info.append({
                'id': idx,
                'name': row['name'],
                'city': row['city'],
                'description': row['description'][:100] if pd.notna(row['description']) else ''
            })
        
        prompt = f"""用户需求：{user_intent['original_query']}

意图分析：
- 兴趣点：{', '.join(user_intent.get('interests', []))}
- 活动：{', '.join(user_intent.get('activities', []))}
- 风格：{user_intent.get('travel_style', '观光游')}

候选景点（前50个）：
{json.dumps(candidates_info, ensure_ascii=False, indent=2)}

请根据用户意图对以上景点重新排序，返回最相关的{topk}个景点的ID列表。
只返回JSON格式的ID列表：[id1, id2, ...]"""
        
        try:
            response = self._call_llm(prompt)
            ranked_ids = json.loads(response)
            
            # 按LLM排序结果重排
            reranked = candidates_df.loc[ranked_ids].copy()
            reranked['llm_rank'] = range(1, len(reranked) + 1)
            
            return reranked.head(topk)
            
        except Exception as e:
            print(f"LLM重排序失败: {e}")
            print("回退到规则模式")
            return self._rule_based_rerank(candidates_df, user_intent, topk)
    
    def _call_llm(self, prompt):
        """调用LLM"""
        if self.llm_model is None:
            raise ValueError("LLM模型未初始化")
        
        # 复用llm_generator中的LLM调用逻辑
        try:
            # 如果llm_model是LLMGenerator实例
            if hasattr(self.llm_model, 'tokenizer') and hasattr(self.llm_model, 'model'):
                messages = [
                    {"role": "system", "content": "你是专业的旅游推荐助手"},
                    {"role": "user", "content": prompt}
                ]
                
                text = self.llm_model.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                model_inputs = self.llm_model.tokenizer([text], return_tensors="pt").to(self.llm_model.model.device)
                
                generated_ids = self.llm_model.model.generate(
                    **model_inputs,
                    max_new_tokens=300,
                    temperature=0.3,
                    do_sample=True
                )
                
                generated_ids = [
                    output_ids[len(input_ids):] 
                    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                
                response = self.llm_model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return response.strip()
            
            # 如果llm_model有generate方法
            elif hasattr(self.llm_model, 'generate'):
                messages = [
                    {"role": "system", "content": "你是专业的旅游推荐助手"},
                    {"role": "user", "content": prompt}
                ]
                response = self.llm_model.generate(messages)
                return response
            
            # 如果llm_model有chat方法（API调用）
            elif hasattr(self.llm_model, 'chat'):
                messages = [
                    {"role": "system", "content": "你是专业的旅游推荐助手"},
                    {"role": "user", "content": prompt}
                ]
                response = self.llm_model.chat(messages)
                return response
            
            else:
                raise ValueError("不支持的LLM模型类型，需要tokenizer/model或generate/chat方法")
                
        except Exception as e:
            raise RuntimeError(f"LLM调用失败: {e}")

if __name__ == "__main__":
    # 测试
    print("="*60)
    print("测试 LLM Reranker")
    print("="*60)
    
    reranker = LLMReranker(use_template=True)
    
    # 构造测试数据
    test_candidates = pd.DataFrame({
        'poi_id': [f'POI_{i:04d}' for i in range(20)],
        'name': ['喀纳斯湖', '禾木村', '天山天池', '赛里木湖', '那拉提草原',
                 '布达拉宫', '纳木错', '羊卓雍措', '大昭寺', '扎什伦布寺',
                 '泸沽湖', '玉龙雪山', '虎跳峡', '香格里拉', '梅里雪山',
                 '九寨沟', '稻城亚丁', '峨眉山', '乐山大佛', '都江堰'],
        'city': ['阿勒泰', '阿勒泰', '乌鲁木齐', '伊犁', '伊犁',
                 '拉萨', '拉萨', '日喀则', '拉萨', '日喀则',
                 '丽江', '丽江', '香格里拉', '香格里拉', '德钦',
                 '阿坝', '甘孜', '乐山', '乐山', '成都'],
        'province': ['新疆']*5 + ['西藏']*5 + ['云南']*5 + ['四川']*5,
        'description': ['湖泊']*20,
        'semantic_score': [0.8 - i*0.02 for i in range(20)],
        'final_score': [0.8 - i*0.02 for i in range(20)]
    })
    
    test_intent = {
        'original_query': '想去新疆看雪山和草原，拍秋天的景色',
        'province': '新疆',
        'interests': ['雪山', '草原'],
        'activities': ['拍照'],
        'season_preference': '秋',
        'travel_style': '摄影游'
    }
    
    reranked = reranker.rerank(test_candidates, test_intent, topk=10)
    
    print(f"\n重排序结果 Top 10:")
    print(reranked[['name', 'city', 'rerank_score']].to_string(index=False))

