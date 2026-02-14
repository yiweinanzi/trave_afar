"""
GoAfar Web UI (Gradio) backed by the unified pipeline.
"""
import argparse
import os
import sys
from datetime import datetime

import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from embedding.vector_builder import ensure_embedding_artifacts, search_similar_pois
from schemas.recommendation import RecommendationRequest, model_to_dict
from service.pipeline import get_pipeline


_PIPELINE = None
_CFG = None


def init_pipeline():
    global _PIPELINE, _CFG
    if _PIPELINE is None:
        _PIPELINE = get_pipeline()
        _CFG = _PIPELINE.config
    return _PIPELINE, _CFG


def _render_recall_rows(df, topn=10):
    if len(df) == 0:
        return "无候选"
    lines = ["| 景点名称 | 城市 | 省份 | 分数 |", "|---|---|---|---|"]
    score_col = "final_score"
    if "rerank_score" in df.columns:
        score_col = "rerank_score"
    elif "semantic_score" in df.columns:
        score_col = "semantic_score"
    for _, row in df.head(topn).iterrows():
        score = float(row.get(score_col, 0.0))
        lines.append(
            f"| {row.get('name', '')} | {row.get('city', '')} | {row.get('province', '')} | {score:.4f} |"
        )
    return "\n".join(lines)


def recommend_route_ui(query, province_choice, max_hours, topk, use_llm_rerank):
    try:
        if not query or not query.strip():
            return ("请输入旅游需求", "", "", "", "")

        pipeline, _ = init_pipeline()
        province = None
        if province_choice not in ("自动识别", "全部"):
            province = province_choice

        req = RecommendationRequest(
            query_text=query.strip(),
            province=province,
            max_hours=float(max_hours),
            topk_candidates=int(topk),
            use_llm=bool(use_llm_rerank),
            return_debug=True,
        )
        resp = pipeline.recommend(req)

        if not resp.success:
            return (
                f"**用户查询**: {query}\n\n❌ {resp.error}",
                "",
                "",
                "",
                "",
            )

        intent = model_to_dict(resp.user_intent) if resp.user_intent else {}
        debug = model_to_dict(resp.debug) if resp.debug else {}

        intent_text = (
            f"**用户查询**: {query}\n\n"
            f"**意图分析**:\n"
            f"- 省份: {intent.get('province') or '未识别'}\n"
            f"- 兴趣: {', '.join(intent.get('interests', []) or ['未识别'])}\n"
            f"- 活动: {', '.join(intent.get('activities', []) or ['未识别'])}\n"
            f"- 天数: {intent.get('duration_days') or '未指定'}\n"
            f"- 风格: {intent.get('travel_style') or '观光游'}\n"
            f"- 检索查询: {debug.get('search_query', query)}\n"
        )

        recall_text = (
            f"**召回分路统计**\n"
            f"- 语义召回命中: {debug.get('recall_sources', {}).get('dense', 0)}\n"
            f"- 行为召回命中: {debug.get('recall_sources', {}).get('behavior', 0)}\n"
            f"- 地理召回命中: {debug.get('recall_sources', {}).get('geo', 0)}\n"
            f"- 重排前候选: {debug.get('candidate_count_before_rerank', 0)}\n"
            f"- 重排后候选: {debug.get('candidate_count_after_rerank', 0)}\n"
        )

        planning_text = (
            f"**规划状态**\n"
            f"- 时间矩阵 provider: {debug.get('matrix_provider', 'unknown')}\n"
            f"- 降级模式: {'是' if resp.degraded else '否'}\n"
            f"- 回退事件: {', '.join(debug.get('fallback_events', []) or ['无'])}\n"
            f"- 访问景点数: {resp.num_pois}\n"
            f"- 总时长: {resp.total_hours:.2f} 小时\n"
        )

        route_lines = [
            f"✨ **标题**: {resp.title}",
            "",
            f"📝 **描述**: {resp.description}",
            "",
            "**行程安排**:",
        ]
        for idx, stop in enumerate(resp.route, 1):
            if idx == 1:
                route_lines.append(f"- 🚩 起点: {stop.poi_name} ({stop.arrival_time_str or '--:--'})")
            elif idx == len(resp.route):
                route_lines.append(f"- 🏁 终点: {stop.poi_name} ({stop.arrival_time_str or '--:--'})")
            else:
                route_lines.append(
                    f"- {idx-1}. {stop.poi_name} | 到达 {stop.arrival_time_str or '--:--'} | 停留 {stop.stay_min} 分钟"
                )
        route_text = "\n".join(route_lines)

        stats_text = (
            f"**统计信息**\n"
            f"- 省份: {resp.province}\n"
            f"- 景点数: {resp.num_pois}\n"
            f"- 总时长: {resp.total_hours:.2f} 小时\n"
            f"- 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        )
        return intent_text, recall_text, planning_text, route_text, stats_text
    except Exception as exc:
        return (f"❌ 错误: {exc}", "", "", "", "")


def search_pois_only(query, province_choice, topk):
    try:
        if not query or not query.strip():
            return "请输入搜索词"
        _, cfg = init_pipeline()
        emb_ok = ensure_embedding_artifacts(
            emb_file=f"{cfg.paths.emb_dir}/poi_emb.npy",
            meta_file=f"{cfg.paths.emb_dir}/poi_meta.csv",
            poi_csv=cfg.paths.poi_csv,
            output_dir=cfg.paths.emb_dir,
            model_path=cfg.embedding.model_path,
            use_gpu=cfg.embedding.use_gpu,
            auto_build=cfg.embedding.auto_build_if_missing,
            build_faiss=(cfg.embedding.backend in {"auto", "faiss"}),
            faiss_index_file=cfg.embedding.faiss_index_file,
        )
        if not emb_ok:
            return "❌ 向量产物缺失，且自动构建失败"

        results = search_similar_pois(
            query_text=query.strip(),
            topk=int(topk),
            emb_file=f"{cfg.paths.emb_dir}/poi_emb.npy",
            meta_file=f"{cfg.paths.emb_dir}/poi_meta.csv",
            model_path=cfg.embedding.model_path,
            use_gpu=cfg.embedding.use_gpu,
            backend=cfg.embedding.backend,
            auto_build=False,
            poi_csv=cfg.paths.poi_csv,
            faiss_index_file=cfg.embedding.faiss_index_file,
        )
        if province_choice not in ("全部", "自动识别"):
            results = results[results["province"] == province_choice]

        table = _render_recall_rows(results, topn=min(int(topk), 15))
        return f"**检索结果** (共 {len(results)} 条)\n\n{table}"
    except Exception as exc:
        return f"❌ 检索失败: {exc}"


def create_ui():
    province_options = ["自动识别", "全部", "新疆", "西藏", "云南", "四川", "甘肃", "青海", "宁夏", "内蒙古"]

    with gr.Blocks(title="GoAfar 智能旅行推荐", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
# GoAfar 智能旅行路线推荐
统一 Pipeline：意图理解 → 多路召回(RRF) → 重排 → 时间矩阵(OSRM/Haversine) → VRPTW → 文案生成
"""
        )

        with gr.Tabs():
            with gr.Tab("完整路线推荐"):
                with gr.Row():
                    with gr.Column(scale=2):
                        query_input = gr.Textbox(label="旅游需求", lines=2, placeholder="例如：想去新疆喀纳斯看秋天的景色，拍照")
                        with gr.Row():
                            province_select = gr.Dropdown(choices=province_options, value="自动识别", label="目标省份")
                            max_hours_slider = gr.Slider(minimum=4, maximum=16, value=10, step=1, label="最大行程时长（小时）")
                        with gr.Row():
                            topk_slider = gr.Slider(minimum=10, maximum=80, value=30, step=5, label="候选POI数量")
                            use_llm_rerank_check = gr.Checkbox(value=True, label="启用LLM增强")
                        recommend_btn = gr.Button("开始推荐", variant="primary")

                with gr.Row():
                    with gr.Column():
                        intent_output = gr.Markdown(label="意图理解")
                        recall_output = gr.Markdown(label="召回结果")
                    with gr.Column():
                        planning_output = gr.Markdown(label="规划状态")
                        route_output = gr.Markdown(label="推荐路线")
                        stats_output = gr.Markdown(label="统计信息")

                recommend_btn.click(
                    fn=recommend_route_ui,
                    inputs=[query_input, province_select, max_hours_slider, topk_slider, use_llm_rerank_check],
                    outputs=[intent_output, recall_output, planning_output, route_output, stats_output],
                )

            with gr.Tab("语义检索"):
                with gr.Row():
                    with gr.Column():
                        search_query = gr.Textbox(label="搜索查询", lines=1, placeholder="例如：雪山、草原、古城")
                        with gr.Row():
                            search_province = gr.Dropdown(choices=province_options, value="全部", label="省份过滤")
                            search_topk = gr.Slider(minimum=5, maximum=30, value=10, step=1, label="返回数量")
                        search_btn = gr.Button("搜索", variant="primary")
                search_output = gr.Markdown(label="检索结果")
                search_btn.click(fn=search_pois_only, inputs=[search_query, search_province, search_topk], outputs=search_output)

            with gr.Tab("系统信息"):
                gr.Markdown(
                    f"""
## 当前状态
- 统一入口: `src/service/pipeline.py`
- 统一配置: `configs/runtime.yaml`
- 多路召回: semantic + behavior + geo + RRF
- 规划层: OSRM 优先，失败自动回退 Haversine
- 更新时间: {datetime.now().strftime("%Y-%m-%d %H:%M")}
"""
                )
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GoAfar Web UI")
    parser.add_argument("--port", type=int, default=7860, help="端口号")
    parser.add_argument("--share", action="store_true", help="生成公网链接")
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="服务器地址")
    args = parser.parse_args()

    demo = create_ui()
    demo.launch(server_name=args.server_name, server_port=args.port, share=args.share, show_error=True)
