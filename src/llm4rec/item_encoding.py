"""
POI编码模块
使用LLM理解POI特征，生成结构化表征
"""
import pandas as pd
import json

class POIEncodingModule:
    """POI特征编码模块"""
    
    def __init__(self, llm_model=None, use_template=True):
        self.llm_model = llm_model
        self.use_template = use_template
    
    def encode_poi_features(self, poi_df):
        """
        为POI生成结构化特征
        
        Args:
            poi_df: POI DataFrame
        
        Returns:
            DataFrame: 增强后的POI特征
        """
        print(f"POI特征编码: {len(poi_df)} 个POI")
        
        if self.use_template:
            return self._template_encoding(poi_df)
        else:
            return self._llm_encoding(poi_df)
    
    def _template_encoding(self, poi_df):
        """基于规则的特征编码"""
        poi_df = poi_df.copy()
        
        # 提取景点类型
        poi_df['poi_type'] = poi_df['name'].apply(self._extract_poi_type)
        
        # 提取难度等级
        poi_df['difficulty'] = poi_df['name'].apply(self._extract_difficulty)
        
        return poi_df
    
    def _extract_poi_type(self, name):
        """提取景点类型"""
        type_keywords = {
            '自然景观': ['山', '湖', '河', '海', '峡谷', '草原', '沙漠', '森林', '冰川'],
            '人文景观': ['寺', '庙', '宫', '城', '镇', '村', '博物馆', '遗址'],
            '交通枢纽': ['机场', '站', '码头'],
            '住宿休闲': ['酒店', '度假', '温泉', '乐园']
        }
        
        for poi_type, keywords in type_keywords.items():
            if any(kw in name for kw in keywords):
                return poi_type
        
        return '其他'
    
    def _extract_difficulty(self, name):
        """提取游览难度"""
        if any(kw in name for kw in ['机场', '市区', '镇']):
            return '简单'
        elif any(kw in name for kw in ['峰', '山口', '冰川', '古道']):
            return '困难'
        else:
            return '中等'
    
    def _llm_encoding(self, poi_df):
        """基于LLM的特征编码"""
        if self.llm_model is None:
            return self._template_encoding(poi_df)
        
        enhanced_df = poi_df.copy()
        
        # 批量处理POI描述
        for idx, row in poi_df.iterrows():
            prompt = f"""分析以下景点的特征：
名称：{row['name']}
描述：{row.get('description', '')}

请返回JSON格式：
{{
    "poi_type": "自然景观/人文景观/交通枢纽/住宿休闲/其他",
    "difficulty": "简单/中等/困难",
    "highlights": ["特色1", "特色2", ...]
}}

只返回JSON，不要其他文字。"""
            
            try:
                # 调用LLM
                if hasattr(self.llm_model, 'tokenizer') and hasattr(self.llm_model, 'model'):
                    messages = [
                        {"role": "system", "content": "你是旅游景点分析助手"},
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
                        temperature=0.3
                    )
                    
                    generated_ids = [
                        output_ids[len(input_ids):] 
                        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                    ]
                    
                    response = self.llm_model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    features = json.loads(response.strip())
                    
                    enhanced_df.loc[idx, 'poi_type'] = features.get('poi_type', '其他')
                    enhanced_df.loc[idx, 'difficulty'] = features.get('difficulty', '中等')
                    enhanced_df.loc[idx, 'highlights'] = str(features.get('highlights', []))
                else:
                    # 失败时使用模板编码
                    enhanced_df.loc[idx, 'poi_type'] = self._extract_poi_type(row['name'])
                    enhanced_df.loc[idx, 'difficulty'] = self._extract_difficulty(row['name'])
            except Exception as e:
                # 失败时使用模板编码
                print(f"POI {idx} LLM编码失败: {e}，使用模板编码")
                enhanced_df.loc[idx, 'poi_type'] = self._extract_poi_type(row['name'])
                enhanced_df.loc[idx, 'difficulty'] = self._extract_difficulty(row['name'])
        
        return enhanced_df

class POIEnhancer:
    """POI描述增强器（使用LLM）"""
    
    @staticmethod
    def enhance_description(poi_name, original_desc, llm_model=None):
        """
        使用LLM增强POI描述
        
        Args:
            poi_name: 景点名称
            original_desc: 原始描述
            llm_model: LLM模型
        
        Returns:
            str: 增强后的描述
        """
        if llm_model is None:
            return original_desc
        
        prompt = f"""请为以下景点生成一段吸引人的描述（100字以内）：

景点名称：{poi_name}
原始描述：{original_desc[:200]}

要求：
- 突出景点特色和亮点
- 使用生动的语言
- 包含感官体验
- 适合旅游推荐

只返回描述文字。"""
        
        try:
            # 调用LLM生成
            if hasattr(llm_model, 'tokenizer') and hasattr(llm_model, 'model'):
                messages = [
                    {"role": "system", "content": "你是旅游文案创作助手"},
                    {"role": "user", "content": prompt}
                ]
                
                text = llm_model.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                model_inputs = llm_model.tokenizer([text], return_tensors="pt").to(llm_model.model.device)
                
                generated_ids = llm_model.model.generate(
                    **model_inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    top_p=0.9
                )
                
                generated_ids = [
                    output_ids[len(input_ids):] 
                    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                
                response = llm_model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return response.strip()
            elif hasattr(llm_model, 'generate'):
                response = llm_model.generate(prompt)
                return response.strip()
            elif hasattr(llm_model, 'chat'):
                messages = [
                    {"role": "system", "content": "你是旅游文案创作助手"},
                    {"role": "user", "content": prompt}
                ]
                response = llm_model.chat(messages)
                return response.strip()
            else:
                return original_desc
        except Exception as e:
            print(f"LLM增强失败: {e}")
            return original_desc

if __name__ == "__main__":
    # 测试
    encoder = POIEncodingModule(use_template=True)
    
    test_df = pd.DataFrame({
        'poi_id': ['001007', '002001', '003041'],
        'name': ['喀纳斯湖', '布达拉宫', '丽江古城'],
        'description': ['湖泊', '宫殿', '古城']
    })
    
    enhanced = encoder.encode_poi_features(test_df)
    print(enhanced[['name', 'poi_type', 'difficulty']])

