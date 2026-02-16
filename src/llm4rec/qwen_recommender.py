"""
Qwen推荐器
基于Qwen3-8B实现LLM增强的旅游推荐
参考: Qwen3/examples/demo/cli_demo.py 和 TALLRec
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import ast
import json
import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, List


def resolve_qwen_model_path(model_name_or_path: str = None) -> str:
    """
    解析Qwen模型路径
    按优先级查找：
    1. 环境变量 GOAFAR_QWEN_MODEL_DIR
    2. models/Qwen3-8B
    3. models/models--Qwen--Qwen3-8B
    4. 默认本地路径字符串
    """
    if model_name_or_path and os.path.exists(model_name_or_path):
        return model_name_or_path

    project_root = Path(__file__).resolve().parents[2]

    # 环境变量
    env_model_dir = os.getenv("GOAFAR_QWEN_MODEL_DIR", "")
    if env_model_dir and os.path.exists(env_model_dir):
        return env_model_dir

    # 标准本地路径
    candidates = [
        project_root / "models" / "Qwen3-8B",
        project_root / "models" / "models--Qwen--Qwen3-8B",
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    # 返回默认本地路径（即使目录暂未下载完整）
    return model_name_or_path or str(project_root / "models" / "Qwen3-8B")


class QwenRecommender:
    """Qwen推荐器 - 用于旅游路线推荐"""
    
    def __init__(
        self,
        model_name_or_path: str = None,
        use_gpu: bool = True,
        use_lora: bool = False,
        lora_path: Optional[str] = None
    ):
        """
        初始化Qwen推荐器

        Args:
            model_name_or_path: 模型路径或名称（默认自动查找）
            use_gpu: 是否使用GPU
            use_lora: 是否使用LoRA适配器
            lora_path: LoRA适配器路径
        """
        self.model_name = model_name_or_path or resolve_qwen_model_path()
        self.use_lora = use_lora
        self.lora_path = lora_path
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.intent_min_free_mb = int(os.getenv("GOAFAR_LLM_INTENT_MIN_FREE_MB", "384"))
        self.rerank_min_free_mb = int(os.getenv("GOAFAR_LLM_RERANK_MIN_FREE_MB", "768"))

        print(f"初始化 Qwen 推荐器...")
        print(f"  模型: {self.model_name}")
        print(f"  设备: {self.device}")
        if use_lora:
            print(f"  LoRA: {lora_path or '未指定'}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )

            # 加载LoRA适配器
            if use_lora and lora_path and os.path.exists(lora_path):
                print(f"  加载LoRA适配器: {lora_path}")
                try:
                    from peft import PeftModel
                    self.model = PeftModel.from_pretrained(self.model, lora_path)
                    print("  ✓ LoRA适配器加载完成")
                except ImportError:
                    print("  ⚠ peft未安装，跳过LoRA加载")
                except Exception as e:
                    print(f"  ⚠ LoRA加载失败: {e}")

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

        if not self._has_enough_cuda_memory(self.intent_min_free_mb):
            free_mb, _ = self._get_cuda_mem_info_mb()
            free_display = "unknown" if free_mb is None else f"{free_mb:.0f}MB"
            print(
                f"LLM意图理解跳过: 可用显存不足({free_display} < {self.intent_min_free_mb}MB)，回退模板"
            )
            return self._fallback_intent(query)

        prompt = f"""请分析以下用户的旅游需求，提取关键信息。

用户查询：{query}

仅输出一个JSON对象，字段使用以下命名：
{{
  "province": "目标省份（新疆/西藏/云南/四川/甘肃/青海/宁夏/内蒙古之一，未知则null）",
  "cities": ["城市列表"],
  "interests": ["兴趣点"],
  "activities": ["活动类型"],
  "duration_days": 期望天数（数字或null）,
  "season": "季节偏好（春/夏/秋/冬或null）",
  "style": "旅行风格（摄影游/深度游/休闲游/亲子游/观光游）",
  "constraints": ["约束条件"],
  "keywords": ["用于检索的关键词"]
}}

不要输出解释、代码块或<think>。"""

        try:
            response = self._generate(prompt, max_new_tokens=128, temperature=0.0)
            payload = self._extract_json_payload(response)
            if not isinstance(payload, dict):
                raise ValueError(f"未解析到JSON对象，raw={self._truncate_for_log(response)}")
            result = self._normalize_intent_payload(payload, query)
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
        if self.model is None or len(pois) > 20:
            # 如果POI太多或模型未加载，使用规则
            return [p['poi_id'] for p in pois[:topk] if p.get('poi_id')]

        if not self._has_enough_cuda_memory(self.rerank_min_free_mb):
            free_mb, _ = self._get_cuda_mem_info_mb()
            free_display = "unknown" if free_mb is None else f"{free_mb:.0f}MB"
            print(
                f"LLM重排序跳过: 可用显存不足({free_display} < {self.rerank_min_free_mb}MB)，回退规则"
            )
            return [p['poi_id'] for p in pois[:topk] if p.get('poi_id')]

        llm_topk = max(1, min(int(topk), len(pois)))

        # 构建POI信息
        poi_info = []
        for idx, poi in enumerate(pois[:10]):  # 限制10个，降低OOM风险
            poi_info.append({
                'id': idx,
                'name': self._safe_text(poi.get('name'), max_len=64),
                'city': self._safe_text(poi.get('city'), max_len=32),
                'description': self._safe_text(poi.get('description'), max_len=48),
            })

        if not poi_info:
            return [p['poi_id'] for p in pois[:llm_topk] if p.get('poi_id')]

        query_text = user_intent.get('original_query') or user_intent.get('expanded_query') or ""
        style = user_intent.get('style') or user_intent.get('travel_style') or '观光游'
        prompt = f"""用户需求：{query_text}

用户意图：
- 省份：{user_intent.get('province', '未指定')}
- 兴趣：{', '.join(user_intent.get('interests', []))}
- 活动：{', '.join(user_intent.get('activities', []))}
- 风格：{style}

候选景点（{len(poi_info)}个）：
{json.dumps(poi_info, ensure_ascii=False)}

请根据用户意图，选出最相关的{llm_topk}个景点，按相关性从高到低排序。
只返回JSON格式：{{"ranked_ids": [id1, id2, ...]}}。不要输出解释、代码块或<think>。"""

        try:
            response = self._generate(prompt, max_new_tokens=64, temperature=0.0)
            payload = self._extract_json_payload(response)
            ranked_ids = []
            if isinstance(payload, dict):
                ranked_ids = payload.get('ranked_ids', [])
            elif isinstance(payload, list):
                ranked_ids = payload

            valid_ids = []
            seen = set()
            for x in ranked_ids:
                try:
                    idx = int(x)
                except Exception:
                    continue
                if 0 <= idx < len(poi_info) and idx not in seen:
                    valid_ids.append(idx)
                    seen.add(idx)

            if not valid_ids:
                valid_ids = list(range(min(llm_topk, len(poi_info))))

            # 转换为poi_id
            ranked_poi_ids = [pois[i].get('poi_id') for i in valid_ids if i < len(pois)]
            ranked_poi_ids = [pid for pid in ranked_poi_ids if pid]
            return ranked_poi_ids[:llm_topk]

        except Exception as e:
            print(f"LLM重排序失败: {e}")
            return [p['poi_id'] for p in pois[:llm_topk] if p.get('poi_id')]
    
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
            payload = self._extract_json_payload(response)
            if isinstance(payload, dict):
                result = payload
                return result
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
            payload = self._extract_json_payload(response)
            if isinstance(payload, dict):
                result = payload
                reasons = result.get('reasons', [])
                return '\n'.join([f"✓ {r}" for r in reasons])
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
            {"role": "system", "content": "你是一位专业的旅游规划助手。输出必须简洁、可解析，不要输出<think>标签。"},
            {"role": "user", "content": prompt}
        ]

        # 应用chat模板，优先禁用思维链输出，避免响应被<think>占满
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        
        # tokenize
        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=3072,
        ).to(self.device)

        if self.device == "cuda":
            self._maybe_empty_cuda_cache()

        generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=(temperature is not None and temperature > 0),
        )
        if temperature is not None and temperature > 0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p
        
        # 生成
        with torch.inference_mode():
            try:
                generated_ids = self.model.generate(
                    **model_inputs,
                    **generate_kwargs,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self._maybe_empty_cuda_cache()
                raise
        
        # 解码（只取新生成的部分）
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if self.device == "cuda":
            self._maybe_empty_cuda_cache()

        return response.strip()

    @staticmethod
    def _extract_json_payload(text):
        if text is None:
            return None
        cleaned = str(text).strip()
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
        cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.replace("```", "").strip()

        if not cleaned:
            return None

        try:
            return json.loads(cleaned)
        except Exception:
            pass

        repaired = cleaned.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
        try:
            literal_obj = ast.literal_eval(repaired)
            if isinstance(literal_obj, (dict, list)):
                return literal_obj
        except Exception:
            pass

        decoder = json.JSONDecoder()
        fallback_obj = None
        for i, ch in enumerate(cleaned):
            if ch not in "{[":
                continue
            try:
                obj, _ = decoder.raw_decode(cleaned[i:])
                if isinstance(obj, dict):
                    return obj
                if fallback_obj is None:
                    fallback_obj = obj
            except Exception:
                continue
        if fallback_obj is not None:
            return fallback_obj

        return QwenRecommender._parse_key_value_payload(cleaned)

    @staticmethod
    def _parse_key_value_payload(text):
        lines = [line.strip() for line in str(text).splitlines() if "：" in line or ":" in line]
        if not lines:
            return None

        result = {}
        for line in lines:
            line = line.lstrip("-* ").strip()
            match = re.match(r'["\']?([A-Za-z0-9_\u4e00-\u9fff]+)["\']?\s*[:：]\s*(.+)$', line)
            if not match:
                continue
            key, value_raw = match.group(1), match.group(2).strip().rstrip(",")
            value = value_raw.strip("\"' ")

            if value.startswith("[") and value.endswith("]"):
                inner = value[1:-1].strip()
                if not inner:
                    parsed_value = []
                else:
                    parsed_value = [part.strip("\"' ") for part in inner.split(",") if part.strip()]
            elif value.lower() in {"null", "none"}:
                parsed_value = None
            else:
                num_match = re.fullmatch(r"[-+]?\d+", value)
                if num_match:
                    parsed_value = int(value)
                else:
                    parsed_value = value

            result[key] = parsed_value

        return result or None

    def _normalize_intent_payload(self, payload: Dict[str, Any], query: str) -> Dict[str, Any]:
        source = dict(payload or {})

        province = self._first_non_empty(
            source.get("province"),
            source.get("destination"),
            source.get("region"),
            source.get("location"),
        )
        cities = self._as_list(self._first_non_empty(source.get("cities"), source.get("city")))
        interests = self._as_list(
            self._first_non_empty(
                source.get("interests"),
                source.get("interest"),
                source.get("preferences"),
                source.get("themes"),
            )
        )
        activities = self._as_list(self._first_non_empty(source.get("activities"), source.get("activity")))
        duration_days = self._to_int(
            self._first_non_empty(
                source.get("duration_days"),
                source.get("days"),
                source.get("duration"),
                source.get("trip_days"),
            )
        )
        season = self._first_non_empty(source.get("season"), source.get("season_preference"))
        style = self._first_non_empty(
            source.get("style"),
            source.get("travel_style"),
            source.get("type"),
            source.get("trip_type"),
        )
        constraints = self._as_list(
            self._first_non_empty(source.get("constraints"), source.get("constraint"), source.get("requirements"))
        )
        keywords = self._as_list(self._first_non_empty(source.get("keywords"), source.get("tags"), source.get("search_terms")))

        if not keywords:
            seed_terms = []
            if province:
                seed_terms.append(province)
            seed_terms.extend(cities[:2])
            seed_terms.extend(interests[:3])
            seed_terms.extend(activities[:2])
            keywords = [term for term in seed_terms if term]

        normalized = dict(source)
        normalized.update(
            {
                "original_query": query,
                "province": province,
                "cities": cities,
                "interests": interests,
                "activities": activities,
                "duration_days": duration_days,
                "season": season,
                "season_preference": season,
                "style": style or "观光游",
                "travel_style": style or "观光游",
                "constraints": constraints,
                "keywords": keywords,
            }
        )

        if not normalized.get("expanded_query"):
            if keywords:
                normalized["expanded_query"] = " ".join(keywords)
            else:
                normalized["expanded_query"] = query

        return normalized

    @staticmethod
    def _first_non_empty(*values):
        for value in values:
            if value is None:
                continue
            if isinstance(value, str):
                text = value.strip()
                if text:
                    return text
                continue
            if isinstance(value, (list, tuple)):
                if value:
                    return value
                continue
            return value
        return None

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, tuple):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        if not text:
            return []
        if "，" in text:
            parts = [part.strip() for part in text.split("，") if part.strip()]
            if parts:
                return parts
        if "," in text:
            parts = [part.strip() for part in text.split(",") if part.strip()]
            if parts:
                return parts
        return [text]

    @staticmethod
    def _to_int(value):
        if value is None:
            return None
        try:
            if isinstance(value, str):
                match = re.search(r"\d+", value)
                if match:
                    return int(match.group())
            return int(float(value))
        except Exception:
            return None

    @staticmethod
    def _safe_text(value, max_len=None):
        if value is None:
            text = ""
        else:
            text = str(value)
        text = re.sub(r"\s+", " ", text).strip()
        if max_len is not None:
            return text[:max_len]
        return text

    def _get_cuda_mem_info_mb(self):
        if self.device != "cuda" or not torch.cuda.is_available():
            return None, None
        try:
            device_index = torch.cuda.current_device()
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            return free_bytes / (1024 * 1024), total_bytes / (1024 * 1024)
        except Exception:
            return None, None

    def _has_enough_cuda_memory(self, min_free_mb: int) -> bool:
        free_mb, _ = self._get_cuda_mem_info_mb()
        if free_mb is None:
            return True
        return free_mb >= float(min_free_mb)

    @staticmethod
    def _truncate_for_log(text, max_len=180):
        raw = str(text or "").replace("\n", " ").strip()
        if len(raw) <= max_len:
            return raw
        return raw[:max_len] + "..."

    @staticmethod
    def _maybe_empty_cuda_cache():
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    
    def _fallback_intent(self, query):
        """后备意图理解（关键词匹配）"""
        from .intent_understanding import IntentUnderstandingModule
        module = IntentUnderstandingModule(use_template=True)
        return module.understand(query)
    
    def _fallback_content(self, route_pois, province, total_hours, query):
        """后备文案生成（模板）"""
        from content_generation.title_generator import generate_title, generate_description
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
        model_name_or_path='models/Qwen3-8B',
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
