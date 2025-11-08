"""
Qwen推荐器
基于Qwen3-8B实现LLM增强的旅游推荐
参考: Qwen3/examples/demo/cli_demo.py 和 TALLRec
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

class QwenRecommender:
    """Qwen推荐器 - 用于旅游路线推荐"""
    
    def __init__(self, model_name_or_path='Qwen/Qwen3-8B', use_gpu=True):
        """
        初始化Qwen推荐器
        
        Args:
            model_name_or_path: 模型路径或名称
            use_gpu: 是否使用GPU
        """
        self.model_name = model_name_or_path
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        print(f"初始化 Qwen 推荐器...")
        print(f"  模型: {model_name_or_path}")
        print(f"  设备: {self.device}")
        
        # 检查本地是否有模型
        import os
        local_model_path = f"/root/autodl-tmp/goafar_project/models/{model_name_or_path.split('/')[-1]}"
        
        if os.path.exists(local_model_path):
            print(f"  使用本地模型: {local_model_path}")
            model_path = local_model_path
        else:
            model_path = model_name_or_path
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            print("✓ 模型加载完成")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("将使用模板模式作为后备方案")
            self.model = None
            self.tokenizer = None
    
    def understand_intent(self, query):
        """
        理解用户旅游意图
        
        Args:
            query: 用户查询
        
        Returns:
            dict: 结构化的意图信息
        """
        if self.model is None:
            return self._fallback_intent(query)
        
        prompt = f"""请分析以下用户的旅游需求，提取关键信息。

用户查询：{query}

请返回JSON格式，包含：
{{
  "province": "目标省份（新疆/西藏/云南/四川/甘肃/青海/宁夏/内蒙古之一，或null）",
  "cities": ["具体城市列表"],
  "interests": ["兴趣点，如：雪山、湖泊、草原、古城等"],
  "activities": ["活动类型，如：拍照、徒步、骑行等"],
  "duration_days": 期望天数（数字或null）,
  "season": "季节偏好（春/夏/秋/冬或null）",
  "style": "旅行风格（摄影游/深度游/休闲游/亲子游等）",
  "constraints": ["约束条件"],
  "keywords": ["关键词列表，用于检索"]
}}

只返回JSON，不要其他内容。"""
        
        try:
            response = self._generate(prompt, max_new_tokens=300, temperature=0.3)
            
            # 提取JSON
            json_match = response
            if '{' in response:
                start = response.index('{')
                end = response.rindex('}') + 1
                json_match = response[start:end]
            
            result = json.loads(json_match)
            result['original_query'] = query
            return result
            
        except Exception as e:
            print(f"LLM意图理解失败: {e}")
            return self._fallback_intent(query)
    
    def rerank_pois(self, pois, user_intent, topk=20):
        """
        基于LLM对POI重排序
        
        Args:
            pois: POI列表（字典列表）
            user_intent: 用户意图
            topk: 返回Top-K
        
        Returns:
            list: 重排序后的POI ID列表
        """
        if self.model is None or len(pois) > 30:
            # 如果POI太多或模型未加载，使用规则
            return [p['poi_id'] for p in pois[:topk]]
        
        # 构建POI信息
        poi_info = []
        for idx, poi in enumerate(pois[:30]):  # 限制30个，避免token过多
            poi_info.append({
                'id': idx,
                'name': poi['name'],
                'city': poi.get('city', ''),
                'description': poi.get('description', '')[:80]
            })
        
        prompt = f"""用户需求：{user_intent['original_query']}

用户意图：
- 省份：{user_intent.get('province', '未指定')}
- 兴趣：{', '.join(user_intent.get('interests', []))}
- 活动：{', '.join(user_intent.get('activities', []))}
- 风格：{user_intent.get('style', '观光游')}

候选景点（{len(poi_info)}个）：
{json.dumps(poi_info, ensure_ascii=False, indent=2)}

请根据用户意图，选出最相关的{topk}个景点，按相关性从高到低排序。
只返回JSON格式的ID列表：{{"ranked_ids": [id1, id2, ...]}}"""
        
        try:
            response = self._generate(prompt, max_new_tokens=200, temperature=0.1)
            
            # 提取JSON
            if '{' in response:
                start = response.index('{')
                end = response.rindex('}') + 1
                json_str = response[start:end]
                result = json.loads(json_str)
                ranked_ids = result.get('ranked_ids', list(range(topk)))
            else:
                ranked_ids = list(range(topk))
            
            # 转换为poi_id
            ranked_poi_ids = [pois[i]['poi_id'] for i in ranked_ids if i < len(pois)]
            return ranked_poi_ids[:topk]
            
        except Exception as e:
            print(f"LLM重排序失败: {e}")
            return [p['poi_id'] for p in pois[:topk]]
    
    def generate_content(self, route_pois, province, total_hours, query):
        """
        生成路线标题和描述
        
        Args:
            route_pois: 路线POI列表
            province: 省份
            total_hours: 总时长
            query: 用户查询
        
        Returns:
            dict: {'title': 标题, 'description': 描述}
        """
        if self.model is None:
            return self._fallback_content(route_pois, province, total_hours, query)
        
        # 提取核心景点
        core_pois = [p['poi_name'] for p in route_pois[1:-1][:5]]
        
        prompt = f"""请为以下旅游路线生成吸引人的标题和描述。

用户需求：{query}
省份：{province}
核心景点：{', '.join(core_pois)}
行程时长：{total_hours:.1f}小时
景点数量：{len(route_pois)-2}个

要求：
1. 标题：20-40字，使用"｜"分隔，体现{province}特色
2. 描述：80-150字，生动的场景描述，融入感官体验

返回JSON格式：
{{
  "title": "标题内容",
  "description": "描述内容"
}}

只返回JSON，不要其他内容。"""
        
        try:
            response = self._generate(prompt, max_new_tokens=300, temperature=0.7)
            
            # 提取JSON
            if '{' in response:
                start = response.index('{')
                end = response.rindex('}') + 1
                json_str = response[start:end]
                result = json.loads(json_str)
                return result
            else:
                return self._fallback_content(route_pois, province, total_hours, query)
                
        except Exception as e:
            print(f"LLM文案生成失败: {e}")
            return self._fallback_content(route_pois, province, total_hours, query)
    
    def explain_recommendation(self, poi, user_intent):
        """
        生成推荐理由
        
        Args:
            poi: POI字典
            user_intent: 用户意图
        
        Returns:
            str: 推荐理由
        """
        if self.model is None:
            return self._fallback_explanation(poi, user_intent)
        
        prompt = f"""为什么推荐这个景点？

景点：{poi['name']}
位置：{poi.get('city', '')}
描述：{poi.get('description', '')[:100]}

用户需求：{user_intent['original_query']}
用户兴趣：{', '.join(user_intent.get('interests', []))}

请生成3条推荐理由，每条20字以内，突出与用户需求的匹配点。
返回JSON：{{"reasons": ["理由1", "理由2", "理由3"]}}"""
        
        try:
            response = self._generate(prompt, max_new_tokens=150, temperature=0.5)
            
            if '{' in response:
                start = response.index('{')
                end = response.rindex('}') + 1
                result = json.loads(response[start:end])
                reasons = result.get('reasons', [])
                return '\n'.join([f"✓ {r}" for r in reasons])
            else:
                return self._fallback_explanation(poi, user_intent)
                
        except Exception as e:
            print(f"LLM解释生成失败: {e}")
            return self._fallback_explanation(poi, user_intent)
    
    def _generate(self, prompt, max_new_tokens=512, temperature=0.7, top_p=0.9):
        """
        调用Qwen生成文本
        
        Args:
            prompt: 提示词
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: nucleus sampling参数
        
        Returns:
            str: 生成的文本
        """
        messages = [
            {"role": "system", "content": "你是一位专业的旅游规划助手，擅长理解用户需求并提供个性化的旅游建议。"},
            {"role": "user", "content": prompt}
        ]
        
        # 应用chat模板
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # tokenize
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        # 生成
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0
            )
        
        # 解码（只取新生成的部分）
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response.strip()
    
    def _fallback_intent(self, query):
        """后备意图理解（关键词匹配）"""
        from .intent_understanding import IntentUnderstandingModule
        module = IntentUnderstandingModule(use_template=True)
        return module.understand(query)
    
    def _fallback_content(self, route_pois, province, total_hours, query):
        """后备文案生成（模板）"""
        from ..content_generation.title_generator import generate_title, generate_description
        return {
            'title': generate_title(route_pois, province, query),
            'description': generate_description(route_pois, province, total_hours, query)
        }
    
    def _fallback_explanation(self, poi, user_intent):
        """后备推荐解释"""
        reasons = []
        interests = user_intent.get('interests', [])
        
        for interest in interests:
            if interest in poi['name'] or (poi.get('description') and interest in poi['description']):
                reasons.append(f"符合您对{interest}的需求")
        
        if not reasons:
            reasons.append("该地区的特色景点")
        
        return '\n'.join([f"✓ {r}" for r in reasons[:3]])

if __name__ == "__main__":
    print("="*60)
    print("测试 Qwen 推荐器")
    print("="*60)
    
    # 初始化（会尝试加载模型，如果失败则用模板）
    recommender = QwenRecommender(
        model_name_or_path='Qwen/Qwen3-8B',
        use_gpu=False  # 改为True如果有GPU
    )
    
    # 测试意图理解
    query = "想去新疆喀纳斯看3天秋天的景色，拍照"
    print(f"\n用户查询: {query}")
    
    intent = recommender.understand_intent(query)
    print(f"\n意图分析:")
    print(json.dumps(intent, ensure_ascii=False, indent=2))
    
    # 测试文案生成
    test_pois = [
        {'poi_name': '乌鲁木齐机场', 'poi_city': '乌鲁木齐'},
        {'poi_name': '喀纳斯湖', 'poi_city': '阿勒泰'},
        {'poi_name': '禾木村', 'poi_city': '阿勒泰'},
        {'poi_name': '白哈巴村', 'poi_city': '阿勒泰'},
        {'poi_name': '乌鲁木齐机场', 'poi_city': '乌鲁木齐'}
    ]
    
    content = recommender.generate_content(test_pois, '新疆', 10.5, query)
    print(f"\n生成文案:")
    print(f"标题: {content['title']}")
    print(f"描述: {content['description']}")

