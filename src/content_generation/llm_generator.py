"""
LLM文案生成器
支持Qwen、Llava等大模型生成高质量文案
"""
import os
import json

class LLMGenerator:
    """大模型文案生成器"""
    
    def __init__(self, model_type='qwen', model_path=None, use_api=False):
        """
        初始化生成器
        
        Args:
            model_type: 模型类型 ('qwen', 'llava', 'api')
            model_path: 本地模型路径
            use_api: 是否使用API（推荐用于生产环境）
        """
        self.model_type = model_type
        self.model_path = model_path
        self.use_api = use_api
        
        if use_api:
            self.model = None
            print("使用API模式（需要配置API密钥）")
        else:
            self._load_model()
    
    def _load_model(self):
        """加载本地模型"""
        if self.model_type == 'qwen':
            self._load_qwen()
        elif self.model_type == 'llava':
            self._load_llava()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def _load_qwen(self):
        """加载Qwen模型"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model_path = self.model_path or "Qwen/Qwen2.5-0.5B-Instruct"
            
            print(f"加载 Qwen 模型: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                cache_dir="/root/autodl-tmp/goafar_project/models"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True,
                cache_dir="/root/autodl-tmp/goafar_project/models"
            )
            print("✓ Qwen模型加载完成")
            
        except Exception as e:
            print(f"Qwen模型加载失败: {e}")
            print("将使用模板生成模式")
            self.model = None
    
    def _load_llava(self):
        """加载Llava模型（多模态）"""
        try:
            print("Llava模型支持正在开发中...")
            print("当前版本使用Qwen文本模型")
            self._load_qwen()
        except Exception as e:
            print(f"Llava模型加载失败: {e}")
            self.model = None
    
    def generate_title(self, route_pois, province, query=None):
        """
        生成路线标题
        
        Args:
            route_pois: POI列表
            province: 省份
            query: 用户查询
        
        Returns:
            str: 生成的标题
        """
        if self.use_api or self.model is None:
            # 使用模板或API
            return self._generate_title_template(route_pois, province, query)
        
        # 使用本地模型生成
        prompt = self._build_title_prompt(route_pois, province, query)
        
        try:
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9
            )
            
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response.strip()
            
        except Exception as e:
            print(f"LLM生成失败，回退到模板: {e}")
            return self._generate_title_template(route_pois, province, query)
    
    def generate_description(self, route_pois, province, total_hours, query=None):
        """
        生成路线描述
        
        Args:
            route_pois: POI列表
            province: 省份
            total_hours: 总时长
            query: 用户查询
        
        Returns:
            str: 生成的描述
        """
        if self.use_api or self.model is None:
            return self._generate_desc_template(route_pois, province, total_hours, query)
        
        prompt = self._build_desc_prompt(route_pois, province, total_hours, query)
        
        try:
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9
            )
            
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response.strip()
            
        except Exception as e:
            print(f"LLM生成失败，回退到模板: {e}")
            return self._generate_desc_template(route_pois, province, total_hours, query)
    
    def _get_system_prompt(self):
        """获取系统提示词"""
        return """你是一位专业的旅游文案撰写专家。请遵循以下风格：

1. 标题风格：
   - 使用"｜"分隔主题和亮点
   - 突出天数和核心景点
   - 使用诗意化表达，避免平铺直叙
   - 示例："喀纳斯秘境3日｜禾木村落星空下，探寻阿尔泰山的秋日童话"

2. 描述风格：
   - 用生动的场景描绘代替干巴巴的介绍
   - 融入感官体验（视觉、听觉、触觉）
   - 突出独特性和记忆点
   - 适度使用emoji增强亲和力

3. 避免：
   - 标题党、夸大宣传
   - 过于笼统的描述
   - 纯粹的信息罗列"""
    
    def _build_title_prompt(self, route_pois, province, query):
        """构建标题生成提示词"""
        poi_names = [p['poi_name'] for p in route_pois[:5]]
        spots = '、'.join(poi_names)
        
        prompt = f"""请为以下{province}旅游路线生成一个吸引人的标题：

用户需求：{query or '未指定'}
省份：{province}
核心景点：{spots}
景点数量：{len(route_pois)-2}个

要求：
- 标题长度20-40字
- 体现{province}特色
- 突出核心景点
- 使用"｜"分隔符

只返回标题，不要其他内容。"""
        
        return prompt
    
    def _build_desc_prompt(self, route_pois, province, total_hours, query):
        """构建描述生成提示词"""
        poi_list = '\n'.join([f"- {p['poi_name']} ({p['poi_city']})" for p in route_pois[:6]])
        
        prompt = f"""请为以下旅游路线生成一段吸引人的推荐描述：

用户需求：{query or '未指定'}
省份：{province}
行程时长：{total_hours:.1f}小时
景点列表：
{poi_list}

要求：
- 描述长度80-150字
- 突出行程亮点
- 融入感官体验
- 体现{province}文化特色

只返回描述文字，不要标题。"""
        
        return prompt
    
    def _generate_title_template(self, route_pois, province, query):
        """使用模板生成标题（回退方案）"""
        from .title_generator import generate_title
        return generate_title(route_pois, province, query)
    
    def _generate_desc_template(self, route_pois, province, total_hours, query):
        """使用模板生成描述（回退方案）"""
        from .title_generator import generate_description
        return generate_description(route_pois, province, total_hours, query)

    def analyze_user_intent(self, query):
        """
        分析用户意图
        
        Args:
            query: 用户查询文本
        
        Returns:
            dict: 意图分析结果
        """
        if self.use_api or self.model is None:
            # 简单的关键词匹配
            return self._analyze_intent_simple(query)
        
        prompt = f"""分析以下用户的旅游需求，提取关键信息：

用户查询：{query}

请以JSON格式返回分析结果，包含：
- province: 目标省份
- interests: 兴趣点列表（如：雪山、湖泊、古城）
- duration_preference: 期望行程天数
- style: 旅行风格（如：深度游、打卡游、摄影游）

只返回JSON，不要其他内容。"""
        
        try:
            messages = [
                {"role": "system", "content": "你是旅游行程规划助手"},
                {"role": "user", "content": prompt}
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=200,
                temperature=0.3
            )
            
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # 尝试解析JSON
            import json
            result = json.loads(response)
            return result
            
        except Exception as e:
            print(f"LLM意图分析失败: {e}")
            return self._analyze_intent_simple(query)
    
    def _analyze_intent_simple(self, query):
        """简单的意图分析（关键词匹配）"""
        provinces = ['新疆', '西藏', '云南', '四川', '甘肃', '青海', '宁夏', '内蒙古']
        interests_keywords = {
            '雪山': ['雪山', '冰川', '雪峰'],
            '湖泊': ['湖', '海子', '错'],
            '草原': ['草原', '牧场'],
            '古城': ['古城', '古镇', '古村'],
            '峡谷': ['峡谷', '沟'],
            '沙漠': ['沙漠', '沙丘']
        }
        
        result = {
            'province': None,
            'interests': [],
            'duration_preference': None,
            'style': '观光游'
        }
        
        # 提取省份
        for prov in provinces:
            if prov in query:
                result['province'] = prov
                break
        
        # 提取兴趣点
        for interest, keywords in interests_keywords.items():
            if any(kw in query for kw in keywords):
                result['interests'].append(interest)
        
        # 提取天数
        import re
        days_match = re.search(r'(\d+)[天日]', query)
        if days_match:
            result['duration_preference'] = int(days_match.group(1))
        
        # 判断风格
        if '拍照' in query or '摄影' in query:
            result['style'] = '摄影游'
        elif '深度' in query or '体验' in query:
            result['style'] = '深度游'
        elif '休闲' in query or '度假' in query:
            result['style'] = '休闲游'
        
        return result

if __name__ == "__main__":
    # 测试
    print("="*60)
    print("测试 LLM 文案生成器")
    print("="*60)
    
    # 使用模板模式测试
    generator = LLMGenerator(model_type='qwen', use_api=False)
    
    # 测试意图分析
    query = "想去新疆喀纳斯看3天秋天的景色，拍照"
    intent = generator.analyze_user_intent(query)
    print(f"\n用户查询: {query}")
    print(f"意图分析: {json.dumps(intent, ensure_ascii=False, indent=2)}")
    
    # 测试文案生成
    test_pois = [
        {'poi_name': '起点', 'poi_city': '乌鲁木齐'},
        {'poi_name': '喀纳斯湖', 'poi_city': '阿勒泰'},
        {'poi_name': '禾木村', 'poi_city': '阿勒泰'},
        {'poi_name': '白哈巴村', 'poi_city': '阿勒泰'},
        {'poi_name': '终点', 'poi_city': '乌鲁木齐'}
    ]
    
    title = generator.generate_title(test_pois, '新疆', query)
    desc = generator.generate_description(test_pois, '新疆', 8.5, query)
    
    print(f"\n生成标题: {title}")
    print(f"生成描述: {desc}")

