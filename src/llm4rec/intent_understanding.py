"""
意图理解模块
使用LLM理解用户旅游需求，提取结构化信息

参考: LLM4Rec的Query Understanding部分
"""
import json
import re

class IntentUnderstandingModule:
    """用户意图理解模块"""
    
    def __init__(self, llm_model=None, use_template=True):
        """
        初始化
        
        Args:
            llm_model: LLM模型实例（Qwen/GPT等）
            use_template: 是否使用模板模式（无需LLM）
        """
        self.llm_model = llm_model
        self.use_template = use_template
        
        if use_template:
            print("意图理解：使用模板模式（关键词匹配）")
        else:
            print("意图理解：使用LLM模式")
    
    def understand(self, query):
        """
        理解用户查询
        
        Args:
            query: 用户查询文本
        
        Returns:
            dict: {
                'province': 目标省份,
                'cities': 具体城市列表,
                'interests': 兴趣点（雪山、湖泊等）,
                'activities': 活动类型（拍照、徒步等）,
                'duration_days': 期望天数,
                'season_preference': 季节偏好,
                'travel_style': 旅行风格,
                'constraints': 约束条件,
                'expanded_query': LLM扩展后的查询
            }
        """
        if self.use_template or self.llm_model is None:
            return self._template_understanding(query)
        else:
            return self._llm_understanding(query)
    
    def _template_understanding(self, query):
        """基于模板的意图理解（关键词匹配）"""
        result = {
            'original_query': query,
            'province': None,
            'cities': [],
            'interests': [],
            'activities': [],
            'duration_days': None,
            'season_preference': None,
            'travel_style': '观光游',
            'constraints': [],
            'expanded_query': query
        }
        
        # 省份映射
        provinces = {
            '新疆': ['新疆', '乌鲁木齐', '喀纳斯', '伊犁', '吐鲁番'],
            '西藏': ['西藏', '拉萨', '布达拉宫', '纳木错', '珠峰'],
            '云南': ['云南', '昆明', '大理', '丽江', '香格里拉'],
            '四川': ['四川', '成都', '九寨沟', '稻城', '峨眉山'],
            '甘肃': ['甘肃', '兰州', '敦煌', '嘉峪关', '张掖'],
            '青海': ['青海', '西宁', '青海湖', '茶卡'],
            '宁夏': ['宁夏', '银川', '沙坡头'],
            '内蒙古': ['内蒙古', '呼伦贝尔', '阿尔山']
        }
        
        # 提取省份
        for prov, keywords in provinces.items():
            if any(kw in query for kw in keywords):
                result['province'] = prov
                break
        
        # 提取兴趣点
        interest_keywords = {
            '雪山': ['雪山', '冰川', '雪峰', '雪域'],
            '湖泊': ['湖', '海子', '错', '池'],
            '草原': ['草原', '牧场', '草地'],
            '古城': ['古城', '古镇', '古村', '老街'],
            '峡谷': ['峡谷', '沟', '谷'],
            '沙漠': ['沙漠', '沙丘', '戈壁'],
            '寺庙': ['寺', '庙', '宫', '塔'],
            '森林': ['森林', '林海', '树林']
        }
        
        for interest, keywords in interest_keywords.items():
            if any(kw in query for kw in keywords):
                result['interests'].append(interest)
        
        # 提取活动
        activity_keywords = {
            '拍照': ['拍照', '摄影', '拍摄', '打卡'],
            '徒步': ['徒步', '登山', '爬山', 'hiking'],
            '骑行': ['骑行', '骑马', '骑车'],
            '自驾': ['自驾', '驾车', '开车'],
            '体验': ['体验', '感受', '了解']
        }
        
        for activity, keywords in activity_keywords.items():
            if any(kw in query for kw in keywords):
                result['activities'].append(activity)
        
        # 提取天数
        days_pattern = re.search(r'(\d+)[天日]', query)
        if days_pattern:
            result['duration_days'] = int(days_pattern.group(1))
        
        # 提取季节
        seasons = ['春', '夏', '秋', '冬']
        for season in seasons:
            if season in query:
                result['season_preference'] = season
                break
        
        # 判断旅行风格
        if '摄影' in query or '拍照' in query:
            result['travel_style'] = '摄影游'
        elif '深度' in query or '体验' in query:
            result['travel_style'] = '深度游'
        elif '休闲' in query or '度假' in query:
            result['travel_style'] = '休闲游'
        elif '亲子' in query or '家庭' in query:
            result['travel_style'] = '亲子游'
        
        # 提取约束
        if '不想' in query or '避免' in query:
            constraint_match = re.search(r'(不想|避免)(.{2,10})', query)
            if constraint_match:
                result['constraints'].append(constraint_match.group(2))
        
        return result
    
    def _llm_understanding(self, query):
        """基于LLM的意图理解"""
        prompt = f"""请分析以下用户的旅游需求，提取关键信息：

用户查询：{query}

请以JSON格式返回，包含：
- province: 目标省份（如果未明确提及则为null）
- cities: 具体城市列表
- interests: 兴趣点列表（如：雪山、湖泊、古城等）
- activities: 活动类型（如：拍照、徒步、骑行等）
- duration_days: 期望行程天数
- season_preference: 季节偏好（春/夏/秋/冬）
- travel_style: 旅行风格（摄影游/深度游/休闲游/亲子游）
- constraints: 约束条件（如：不想太累、预算有限等）
- implicit_needs: 隐含需求（推断出的未明说的需求）

只返回JSON，不要其他文字。"""
        
        try:
            # 调用LLM
            response = self._call_llm(prompt)
            result = json.loads(response)
            result['original_query'] = query
            result['expanded_query'] = self._expand_query(result)
            return result
        except Exception as e:
            print(f"LLM意图理解失败，回退到模板: {e}")
            return self._template_understanding(query)
    
    def _call_llm(self, prompt):
        """调用LLM模型"""
        if self.llm_model is None:
            raise ValueError("LLM模型未初始化")
        
        # 复用llm_generator中的LLM调用逻辑
        try:
            # 如果llm_model是LLMGenerator实例
            if hasattr(self.llm_model, 'tokenizer') and hasattr(self.llm_model, 'model'):
                messages = [
                    {"role": "system", "content": "你是专业的旅游规划助手"},
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
                    max_new_tokens=200,
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
                    {"role": "system", "content": "你是专业的旅游规划助手"},
                    {"role": "user", "content": prompt}
                ]
                response = self.llm_model.generate(messages)
                return response
            
            # 如果llm_model有chat方法（API调用）
            elif hasattr(self.llm_model, 'chat'):
                messages = [
                    {"role": "system", "content": "你是专业的旅游规划助手"},
                    {"role": "user", "content": prompt}
                ]
                response = self.llm_model.chat(messages)
                return response
            
            else:
                raise ValueError("不支持的LLM模型类型，需要tokenizer/model或generate/chat方法")
                
        except Exception as e:
            raise RuntimeError(f"LLM调用失败: {e}")
    
    def _expand_query(self, intent_result):
        """基于意图结果扩展查询"""
        parts = [intent_result['original_query']]
        
        if intent_result.get('season_preference'):
            parts.append(f"{intent_result['season_preference']}季")
        
        if intent_result.get('interests'):
            parts.append('、'.join(intent_result['interests']))
        
        if intent_result.get('activities'):
            parts.append('、'.join(intent_result['activities']))
        
        return ' '.join(parts)

if __name__ == "__main__":
    # 测试
    print("="*60)
    print("测试意图理解模块")
    print("="*60)
    
    module = IntentUnderstandingModule(use_template=True)
    
    test_queries = [
        "想去新疆喀纳斯看3天秋天的景色，拍照",
        "计划去西藏拉萨朝拜布达拉宫，5天深度游",
        "云南大理洱海骑行2天，轻松休闲",
        "四川九寨沟亲子游，适合孩子，不要太累"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        result = module.understand(query)
        print(f"结果: {json.dumps(result, ensure_ascii=False, indent=2)}")

