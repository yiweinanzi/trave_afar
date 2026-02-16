"""
Unified recommendation pipeline used by CLI/Web/API entrypoints.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional
import pandas as pd

# 兼容两种导入方式：
# 1) src.service.pipeline (推荐，避免与 tests 下同名包冲突)
# 2) service.pipeline (legacy，src 在 PYTHONPATH)
try:
    from src.content_generation.title_generator import generate_description, generate_title
    from src.embedding.vector_builder import ensure_embedding_artifacts
    from src.llm4rec.intent_understanding import IntentUnderstandingModule
    from src.llm4rec.llm_reranker import LLMReranker
    from src.llm4rec.qwen_recommender import QwenRecommender
    from src.recommendation.candidate_merger import merge_candidates
    from src.reranking.qwen_reranker import QwenReranker
    from src.routing.time_matrix_builder import build_time_matrix
    from src.utils.data_alignment import summarize_dataset_embedding_alignment
    from src.schemas.recommendation import (
        IntentInfo,
        RecommendationDebug,
        RecommendationRequest,
        RecommendationResponse,
        RouteStop,
    )
except ModuleNotFoundError as exc:
    missing_name = getattr(exc, "name", "")
    if not (missing_name == "src" or missing_name.startswith("src.")):
        raise
    from content_generation.title_generator import generate_description, generate_title
    from embedding.vector_builder import ensure_embedding_artifacts
    from llm4rec.intent_understanding import IntentUnderstandingModule
    from llm4rec.llm_reranker import LLMReranker
    from llm4rec.qwen_recommender import QwenRecommender
    from recommendation.candidate_merger import merge_candidates
    from reranking.qwen_reranker import QwenReranker
    from routing.time_matrix_builder import build_time_matrix
    from utils.data_alignment import summarize_dataset_embedding_alignment
    from schemas.recommendation import (
        IntentInfo,
        RecommendationDebug,
        RecommendationRequest,
        RecommendationResponse,
        RouteStop,
    )
from .config import RuntimeConfig, load_runtime_config


def _load_vrptw_solver_cls():
    try:
        from src.routing.vrptw_solver import VRPTWSolver
        return VRPTWSolver
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", "")
        if not (missing_name == "src" or missing_name.startswith("src.")):
            raise
    from routing.vrptw_solver import VRPTWSolver
    return VRPTWSolver


class RecommendationPipeline:
    def __init__(self, config: RuntimeConfig | None = None):
        self.config = config or load_runtime_config()
        self._intent_template = IntentUnderstandingModule(use_template=True)
        self._reranker_template = LLMReranker(use_template=True)
        self._qwen: Optional[QwenRecommender] = None
        self._qwen_reranker: Optional[QwenReranker] = None
        self._embedding_model: Optional[Any] = None
        self._embedding_ready = False
        self._warmup_messages: list[str] = []

        # 模型状态跟踪
        self._model_status: Dict[str, bool] = {
            'embedding': False,
            'reranker': False,
            'llm': False,
        }

    def warmup(self) -> None:
        cfg = self.config
        self._warmup_messages = []

        poi_path = cfg.resolve_path(cfg.paths.poi_csv)
        if not poi_path.exists():
            raise FileNotFoundError(f"缺少 POI 数据: {poi_path}")

        emb_file = cfg.resolve_path(f"{cfg.paths.emb_dir}/poi_emb.npy")
        meta_file = cfg.resolve_path(f"{cfg.paths.emb_dir}/poi_meta.csv")
        self._embedding_ready = ensure_embedding_artifacts(
            emb_file=str(emb_file),
            meta_file=str(meta_file),
            poi_csv=cfg.paths.poi_csv,
            output_dir=cfg.paths.emb_dir,
            model_path=cfg.embedding.model_path,
            use_gpu=cfg.embedding.use_gpu,
            auto_build=cfg.embedding.auto_build_if_missing,
            build_faiss=(cfg.embedding.backend in {"auto", "faiss"}),
            faiss_index_file=cfg.embedding.faiss_index_file,
        )

        # 训练数据与 embedding 对齐检查（仅告警，不中断服务）
        alignment = summarize_dataset_embedding_alignment(
            dataset_dir=self.config.resolve_path("outputs/datasets"),
            embedding_meta_csv=meta_file,
        )
        if alignment.get("checked") and alignment.get("missing_in_embeddings", 0) > 0:
            missing = alignment["missing_in_embeddings"]
            self._warmup_messages.append(f"training_embedding_mismatch:{missing}")

        if not self._embedding_ready:
            self._warmup_messages.append("embedding_missing")
            return

    def _maybe_get_qwen(self) -> QwenRecommender:
        if self._qwen is None:
            lora_path = None
            if self.config.llm.use_lora and self.config.llm.lora_path:
                lora_path = str(self.config.resolve_path(self.config.llm.lora_path))
            self._qwen = QwenRecommender(
                model_name_or_path=self.config.llm.qwen_model,
                use_gpu=self.config.llm.use_gpu,
                use_lora=self.config.llm.use_lora,
                lora_path=lora_path,
            )
        return self._qwen

    def _maybe_get_qwen_reranker(self) -> Optional[QwenReranker]:
        """获取或初始化QwenReranker实例"""
        if not self.config.rerank.use_reranker_model:
            return None

        if self._qwen_reranker is None:
            try:
                self._qwen_reranker = QwenReranker(
                    model_path=self.config.rerank.qwen_reranker_path,
                    use_gpu=self.config.llm.use_gpu,
                )
                # 检查模型是否成功加载
                if self._qwen_reranker.model is None:
                    self._warmup_messages.append("qwen_reranker_model_unavailable")
                    self._model_status['reranker'] = False
                    return None
                self._model_status['reranker'] = True
            except Exception as exc:
                self._warmup_messages.append(f"qwen_reranker_init_failed:{exc}")
                self._model_status['reranker'] = False
                return None

        return self._qwen_reranker

    @staticmethod
    def _to_intent_info(intent_dict: Dict[str, Any], query: str) -> IntentInfo:
        known_keys = set(IntentInfo.model_fields.keys())
        payload = {}
        extra = {}
        for key, value in intent_dict.items():
            if key in known_keys:
                payload[key] = value
            else:
                extra[key] = value
        if "original_query" not in payload:
            payload["original_query"] = query
        payload["extra"] = extra
        return IntentInfo(**payload)

    def recommend(self, request: RecommendationRequest | Dict[str, Any]) -> RecommendationResponse:
        if isinstance(request, dict):
            request = RecommendationRequest(**request)

        if not self._embedding_ready and self.config.embedding.auto_build_if_missing:
            self.warmup()

        debug = RecommendationDebug(fallback_events=list(self._warmup_messages))

        try:
            query = request.query_text
            province = request.province
            search_query = query
            intent_dict: Dict[str, Any] = {
                "original_query": query,
                "province": province,
                "interests": [],
                "activities": [],
                "expanded_query": query,
            }

            # Step 1: intent understanding
            if request.use_llm:
                if self.config.llm.enabled:
                    qwen = self._maybe_get_qwen()
                    intent_dict = qwen.understand_intent(query)
                else:
                    intent_dict = self._intent_template.understand(query)

                province = province or intent_dict.get("province")
                if intent_dict.get("keywords"):
                    search_query = " ".join(intent_dict.get("keywords", []))
                else:
                    search_query = intent_dict.get("expanded_query", query) or query

            debug.search_query = search_query

            # Step 2: multi-recall
            candidates = merge_candidates(
                query_text=search_query,
                user_id=request.user_id,
                topk_dense=self.config.recall.semantic_topk,
                topk_seq=self.config.recall.behavior_topk,
                topk_geo=self.config.recall.geo_topk,
                province_filter=province,
                poi_csv=self.config.paths.poi_csv,
                user_events_csv=self.config.paths.user_events_csv,
                emb_file=f"{self.config.paths.emb_dir}/poi_emb.npy",
                meta_file=f"{self.config.paths.emb_dir}/poi_meta.csv",
                model_path=self.config.embedding.model_path,
                use_gpu=self.config.embedding.use_gpu,
                backend=self.config.embedding.backend,
                allow_without_embeddings=self.config.fallback.allow_without_embeddings,
                fusion=self.config.recall.fusion,
                rrf_k=self.config.recall.rrf_k,
                dense_weight=self.config.recall.dense_weight,
                behavior_weight=self.config.recall.behavior_weight,
                geo_weight=self.config.recall.geo_weight,
                calibrate=self.config.recall.calibrate,
                final_topk=max(self.config.recall.final_topk, request.topk_candidates),
                use_recbole=self.config.recall.behavior_provider.lower() == "recbole",
                recbole_model_path=self.config.recall.recbole_model_path,
                recbole_config=self.config.recall.recbole_config,
                recbole_use_gpu=self.config.recall.recbole_use_gpu,
                adaptive_fusion_enabled=self.config.recall.adaptive_fusion,
            )
            if not self._embedding_ready and self.config.fallback.allow_without_embeddings:
                debug.fallback_events.append("semantic_recall_unavailable")

            debug.recall_sources = {
                "dense": int(candidates["from_dense"].sum()) if "from_dense" in candidates.columns else 0,
                "behavior": int(candidates["from_behavior"].sum()) if "from_behavior" in candidates.columns else 0,
                "geo": int(candidates["from_geo"].sum()) if "from_geo" in candidates.columns else 0,
            }
            debug.candidate_count_before_rerank = len(candidates)

            if len(candidates) == 0:
                return RecommendationResponse.error_response(query, "未找到匹配的候选景点")

            # Step 3: rerank
            if self.config.rerank.enabled:
                # 3.1: 首先使用LLM进行初步重排（如果启用）
                if request.use_llm and self.config.llm.enabled:
                    try:
                        qwen = self._maybe_get_qwen()
                        topk_for_llm = min(len(candidates), max(10, request.topk_candidates * 2))
                        candidate_list = candidates.head(topk_for_llm).to_dict("records")
                        ranked_ids = qwen.rerank_pois(candidate_list, intent_dict, topk=request.topk_candidates)
                        ranked_map = {pid: i for i, pid in enumerate(ranked_ids)}
                        candidates = candidates[candidates["poi_id"].isin(ranked_map.keys())].copy()
                        candidates["llm_rank"] = candidates["poi_id"].map(ranked_map)
                        candidates = candidates.sort_values("llm_rank").head(request.topk_candidates)
                    except Exception as exc:
                        debug.fallback_events.append(f"llm_rerank_fallback:{exc}")
                        candidates = self._reranker_template.rerank(candidates, intent_dict, topk=request.topk_candidates)
                else:
                    candidates = self._reranker_template.rerank(candidates, intent_dict, topk=request.topk_candidates)

                # 3.2: 使用Qwen3-Reranker-4B进行精排（如果启用且可用）
                if self.config.rerank.use_reranker_model and len(candidates) > 0:
                    try:
                        qwen_reranker = self._maybe_get_qwen_reranker()
                        if qwen_reranker is not None and qwen_reranker.model is not None:
                            # 准备候选列表
                            candidate_list = candidates.to_dict("records")
                            rerank_topk = min(len(candidates), self.config.rerank.rerank_topk)

                            # 使用Reranker模型进行精排
                            reranked = qwen_reranker.rerank(
                                query=query,
                                candidates=candidate_list,
                                topk=rerank_topk
                            )

                            # 将结果转换回DataFrame
                            reranked_ids = [item["poi_id"] for item in reranked]
                            reranked_map = {pid: i for i, pid in enumerate(reranked_ids)}

                            # 过滤并重排
                            candidates = candidates[candidates["poi_id"].isin(reranked_map.keys())].copy()
                            candidates["reranker_rank"] = candidates["poi_id"].map(reranked_map)
                            candidates = candidates.sort_values("reranker_rank").head(rerank_topk)

                            debug.fallback_events.append("qwen_reranker_applied")
                    except Exception as exc:
                        debug.fallback_events.append(f"qwen_reranker_failed:{exc}")
                        # 继续使用已有的LLM重排结果
            else:
                candidates = candidates.head(request.topk_candidates)

            debug.candidate_count_after_rerank = len(candidates)
            if len(candidates) < self.config.fallback.min_candidates_for_planning:
                return RecommendationResponse.error_response(query, "候选景点不足，无法进行路线规划")

            # Step 4: matrix + routing
            planning_candidates = candidates
            if {"lat", "lon"}.issubset(candidates.columns):
                lat = pd.to_numeric(candidates["lat"], errors="coerce")
                lon = pd.to_numeric(candidates["lon"], errors="coerce")
                valid_coord = lat.notna() & lon.notna()
                dropped = int((~valid_coord).sum())
                if dropped > 0:
                    debug.fallback_events.append(f"dropped_invalid_coord_candidates:{dropped}")
                planning_candidates = candidates.loc[valid_coord].copy()

            if len(planning_candidates) < self.config.fallback.min_candidates_for_planning:
                return RecommendationResponse.error_response(query, "候选景点坐标缺失，无法进行路线规划")

            matrix, poi_df, provider_used = build_time_matrix(
                poi_csv=self.config.paths.poi_csv,
                output_path=f"{self.config.paths.routing_dir}/time_matrix.npy",
                avg_speed_kmh=self.config.planner.avg_speed_kmh,
                poi_ids=planning_candidates["poi_id"].tolist(),
                provider=self.config.planner.matrix_provider,
                osrm_url=self.config.planner.osrm_url,
                use_cache=True,
                cache_dir=self.config.paths.cache_dir,
                cache_ttl_hours=self.config.planner.matrix_cache_ttl_hours,
                return_provider=True,
            )
            debug.matrix_provider = str(provider_used)
            if self.config.planner.matrix_provider in {"auto", "osrm"} and str(provider_used) != "osrm":
                debug.fallback_events.append("time_matrix_fallback_to_haversine")

            try:
                solver_cls = _load_vrptw_solver_cls()
            except Exception as exc:
                return RecommendationResponse.error_response(query, f"routing_backend_unavailable: {exc}")

            solver = solver_cls(poi_df, matrix, start_time_min=self.config.planner.start_time_min)
            solution = solver.solve(
                depot_index=0,
                max_duration_hours=min(request.max_hours, self.config.planner.max_duration_hours),
                time_limit_seconds=self.config.planner.solve_time_limit_seconds,
            )
            if not solution:
                return RecommendationResponse.error_response(query, "未找到满足时间窗约束的可行路线")

            # Step 5: content generation
            route_pois = solution["routes"][0]
            if len(route_pois) <= 2 or int(solution.get("visited_pois", 0)) <= 0:
                return RecommendationResponse.error_response(query, "未找到有效游玩景点，请尝试放宽时长或条件")
            province_name = province or (
                planning_candidates.iloc[0]["province"]
                if "province" in planning_candidates.columns
                else None
            )

            if request.use_llm and self.config.llm.enabled:
                try:
                    qwen = self._maybe_get_qwen()
                    content = qwen.generate_content(
                        route_pois=route_pois,
                        province=province_name or "",
                        total_hours=solution["total_time_hours"],
                        query=query,
                    )
                    title = content.get("title", "") or generate_title(route_pois, province_name or "", query)
                    description = content.get("description", "") or generate_description(
                        route_pois, province_name or "", solution["total_time_hours"], query
                    )
                except Exception as exc:
                    debug.fallback_events.append(f"llm_content_fallback:{exc}")
                    title = generate_title(route_pois, province_name or "", query)
                    description = generate_description(route_pois, province_name or "", solution["total_time_hours"], query)
            else:
                title = generate_title(route_pois, province_name or "", query)
                description = generate_description(route_pois, province_name or "", solution["total_time_hours"], query)

            intent_info = self._to_intent_info(intent_dict, query) if request.use_llm else None
            route = [RouteStop(**stop) for stop in route_pois]

            return RecommendationResponse(
                success=True,
                title=title,
                description=description,
                route=route,
                total_hours=float(solution.get("total_time_hours", 0.0)),
                num_pois=int(solution.get("visited_pois", 0)),
                query=query,
                province=province_name,
                user_intent=intent_info,
                degraded=len(debug.fallback_events) > 0,
                debug=debug if request.return_debug else None,
            )
        except Exception as exc:
            return RecommendationResponse.error_response(request.query_text, f"pipeline_error: {exc}")

    def check_model_health(self) -> Dict[str, bool]:
        """
        检查所有模型健康状态

        Returns:
            Dict[str, bool]: 包含各模型健康状态的字典
                - embedding: Embedding模型是否可用
                - reranker: Reranker模型是否可用
                - llm: LLM模型是否可用
                - all: 所有模型是否都可用
        """
        health = {
            'embedding': self._check_embedding_health(),
            'reranker': self._check_reranker_health(),
            'llm': self._check_llm_health(),
        }
        health['all'] = all(health.values())
        return health

    def _check_embedding_health(self) -> bool:
        """检查Embedding模型健康状态"""
        # 检查embedding artifacts
        emb_file = self.config.resolve_path(f"{self.config.paths.emb_dir}/poi_emb.npy")
        meta_file = self.config.resolve_path(f"{self.config.paths.emb_dir}/poi_meta.csv")

        if not (emb_file.exists() and meta_file.exists()):
            return False

        # 检查配置中的模型路径
        model_path = self.config.resolve_path(self.config.embedding.model_path)
        if not model_path.exists():
            return False

        # 检查Qwen3Embedding类是否可用
        try:
            try:
                from src.embedding.qwen3_encoder import Qwen3Embedding
            except ImportError:
                from embedding.qwen3_encoder import Qwen3Embedding
            # 尝试初始化但不加载完整模型
            test_encoder = Qwen3Embedding(
                model_path=str(model_path),
                use_gpu=False,
                max_retries=1
            )
            is_available = test_encoder.is_available()
            if is_available:
                test_encoder.cleanup()
            return is_available
        except Exception:
            return False

    def _check_reranker_health(self) -> bool:
        """检查Reranker模型健康状态"""
        if not self.config.rerank.use_reranker_model:
            return True  # 未启用，视为健康

        # 检查配置中的模型路径
        model_path = self.config.resolve_path(self.config.rerank.qwen_reranker_path)
        if not model_path.exists():
            return False

        # 如果已加载，直接检查
        if self._qwen_reranker is not None:
            return self._qwen_reranker.model is not None

        # 否则尝试加载
        try:
            try:
                from src.reranking.qwen_reranker import QwenReranker
            except ImportError:
                from reranking.qwen_reranker import QwenReranker
            test_reranker = QwenReranker(
                model_path=str(model_path),
                use_gpu=False
            )
            is_available = test_reranker.model is not None
            return is_available
        except Exception:
            return False

    def _check_llm_health(self) -> bool:
        """检查LLM模型健康状态"""
        if not self.config.llm.enabled:
            return True  # 未启用，视为健康

        # 检查配置中的模型路径
        model_path = self.config.resolve_path(self.config.llm.qwen_model)
        if not model_path.exists():
            return False

        use_lora = bool(self.config.llm.use_lora)
        lora_path = None
        if use_lora:
            if not self.config.llm.lora_path:
                return False
            resolved_lora_path = self.config.resolve_path(self.config.llm.lora_path)
            if not resolved_lora_path.exists():
                return False
            lora_path = str(resolved_lora_path)

        # 如果已加载，直接检查
        if self._qwen is not None:
            return self._qwen.model is not None

        # 否则尝试加载
        try:
            try:
                from src.llm4rec.qwen_recommender import QwenRecommender
            except ImportError:
                from llm4rec.qwen_recommender import QwenRecommender
            test_qwen = QwenRecommender(
                model_name_or_path=str(model_path),
                use_gpu=False,
                use_lora=use_lora,
                lora_path=lora_path,
            )
            is_available = test_qwen.model is not None
            return is_available
        except Exception:
            return False

    def cleanup_models(self) -> None:
        """清理所有已加载的模型，释放内存"""
        logger = self._get_logger()

        # 清理LLM模型
        if self._qwen is not None:
            try:
                del self._qwen.model
                del self._qwen.tokenizer
                self._qwen.model = None
                self._qwen.tokenizer = None
            except Exception as e:
                logger.warning(f"清理LLM模型时出错: {e}")
            finally:
                self._qwen = None

        # 清理Reranker模型
        if self._qwen_reranker is not None:
            try:
                del self._qwen_reranker.model
                del self._qwen_reranker.tokenizer
                self._qwen_reranker.model = None
                self._qwen_reranker.tokenizer = None
            except Exception as e:
                logger.warning(f"清理Reranker模型时出错: {e}")
            finally:
                self._qwen_reranker = None

        # 清理Embedding模型
        if self._embedding_model is not None:
            try:
                if hasattr(self._embedding_model, 'cleanup'):
                    self._embedding_model.cleanup()
                else:
                    del self._embedding_model
            except Exception as e:
                logger.warning(f"清理Embedding模型时出错: {e}")
            finally:
                self._embedding_model = None

        # 清理GPU缓存
        self._clear_gpu_cache()

        # 更新状态
        self._model_status = {
            'embedding': False,
            'reranker': False,
            'llm': False,
        }

        logger.info("所有模型已清理")

    def unload_unused_models(self, keep: list = None) -> None:
        """
        卸载未使用的模型，保留指定的模型

        Args:
            keep: 要保留的模型列表，可选: ['embedding', 'reranker', 'llm']
        """
        if keep is None:
            keep = []

        logger = self._get_logger()

        # 清理LLM模型（如果不保留）
        if 'llm' not in keep and self._qwen is not None:
            try:
                del self._qwen.model
                del self._qwen.tokenizer
                self._qwen.model = None
                self._qwen.tokenizer = None
                self._qwen = None
                self._model_status['llm'] = False
                logger.info("LLM模型已卸载")
            except Exception as e:
                logger.warning(f"卸载LLM模型时出错: {e}")

        # 清理Reranker模型（如果不保留）
        if 'reranker' not in keep and self._qwen_reranker is not None:
            try:
                del self._qwen_reranker.model
                del self._qwen_reranker.tokenizer
                self._qwen_reranker.model = None
                self._qwen_reranker.tokenizer = None
                self._qwen_reranker = None
                self._model_status['reranker'] = False
                logger.info("Reranker模型已卸载")
            except Exception as e:
                logger.warning(f"卸载Reranker模型时出错: {e}")

        # 清理Embedding模型（如果不保留）
        if 'embedding' not in keep and self._embedding_model is not None:
            try:
                if hasattr(self._embedding_model, 'cleanup'):
                    self._embedding_model.cleanup()
                else:
                    del self._embedding_model
                self._embedding_model = None
                self._model_status['embedding'] = False
                logger.info("Embedding模型已卸载")
            except Exception as e:
                logger.warning(f"卸载Embedding模型时出错: {e}")

        # 部分清理GPU缓存
        self._clear_gpu_cache()

    def _clear_gpu_cache(self) -> None:
        """清理GPU缓存"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

    def _get_logger(self):
        """获取日志记录器"""
        import logging
        return logging.getLogger(__name__)

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取当前模型信息

        Returns:
            Dict[str, Any]: 包含各模型信息的字典
        """
        info = {
            'embedding': {
                'loaded': self._embedding_model is not None,
                'path': str(self.config.embedding.model_path),
                'healthy': self._model_status['embedding'],
            },
            'reranker': {
                'loaded': self._qwen_reranker is not None,
                'path': str(self.config.rerank.qwen_reranker_path),
                'healthy': self._model_status['reranker'],
                'enabled': self.config.rerank.use_reranker_model,
            },
            'llm': {
                'loaded': self._qwen is not None,
                'path': str(self.config.llm.qwen_model),
                'healthy': self._model_status['llm'],
                'enabled': self.config.llm.enabled,
                'use_lora': self.config.llm.use_lora,
                'lora_path': str(self.config.llm.lora_path) if self.config.llm.lora_path else None,
            },
        }
        return info


_PIPELINE: Optional[RecommendationPipeline] = None


def get_pipeline(config: RuntimeConfig | None = None) -> RecommendationPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = RecommendationPipeline(config=config)
        _PIPELINE.warmup()
    return _PIPELINE


def recommend_route(request: RecommendationRequest | Dict[str, Any]) -> RecommendationResponse:
    pipeline = get_pipeline()
    return pipeline.recommend(request)


def config_to_dict(config: RuntimeConfig) -> Dict[str, Any]:
    return asdict(config)
