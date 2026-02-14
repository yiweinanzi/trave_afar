"""
Unified recommendation pipeline used by CLI/Web/API entrypoints.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

from content_generation.title_generator import generate_description, generate_title
from embedding.vector_builder import ensure_embedding_artifacts
from llm4rec.intent_understanding import IntentUnderstandingModule
from llm4rec.llm_reranker import LLMReranker
from llm4rec.qwen_recommender import QwenRecommender
from recommendation.candidate_merger import merge_candidates
from reranking.qwen_reranker import QwenReranker
from routing.time_matrix_builder import build_time_matrix
from routing.vrptw_solver import VRPTWSolver
from schemas.recommendation import (
    IntentInfo,
    RecommendationDebug,
    RecommendationRequest,
    RecommendationResponse,
    RouteStop,
)
from service.config import RuntimeConfig, load_runtime_config


class RecommendationPipeline:
    def __init__(self, config: RuntimeConfig | None = None):
        self.config = config or load_runtime_config()
        self._intent_template = IntentUnderstandingModule(use_template=True)
        self._reranker_template = LLMReranker(use_template=True)
        self._qwen: Optional[QwenRecommender] = None
        self._qwen_reranker: Optional[QwenReranker] = None
        self._embedding_ready = False
        self._warmup_messages: list[str] = []

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
        if not self._embedding_ready:
            self._warmup_messages.append("embedding_missing")

    def _maybe_get_qwen(self) -> QwenRecommender:
        if self._qwen is None:
            self._qwen = QwenRecommender(
                model_name_or_path=self.config.llm.qwen_model,
                use_gpu=self.config.llm.use_gpu,
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
                    return None
            except Exception as exc:
                self._warmup_messages.append(f"qwen_reranker_init_failed:{exc}")
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
            matrix, poi_df, provider_used = build_time_matrix(
                poi_csv=self.config.paths.poi_csv,
                output_path=f"{self.config.paths.routing_dir}/time_matrix.npy",
                avg_speed_kmh=self.config.planner.avg_speed_kmh,
                poi_ids=candidates["poi_id"].tolist(),
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

            solver = VRPTWSolver(poi_df, matrix, start_time_min=self.config.planner.start_time_min)
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
            province_name = province or (candidates.iloc[0]["province"] if "province" in candidates.columns else None)

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
