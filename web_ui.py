"""
GoAfar Gradio Web UI - 简化版
在线测试推荐系统
"""
import gradio as gr
import sys
import os
sys.path.insert(0, 'src')

import pandas as pd
import json

def test_semantic_search(query, topk=10):
    """测试语义检索"""
    try:
        # 检查向量文件
        if not os.path.exists('outputs/emb/poi_emb.npy'):
            return "⚠️ POI向量未生成。请先运行: `python src/embedding/build_embeddings_gpu.py`"
        
        from embedding.vector_builder import search_similar_pois
        
        results = search_similar_pois(query, topk=topk, use_gpu=False)
        
        # 格式化输出
        output = f"## 检索结果 (Top {min(topk, len(results))})\n\n"
        
        for i, (_, row) in enumerate(results.head(topk).iterrows(), 1):
            output += f"### {i}. {row['name']}\n"
            output += f"- **省份**: {row['province']}\n"
            output += f"- **城市**: {row['city']}\n"
            output += f"- **相似度**: {row['semantic_score']:.4f}\n"
            output += f"- **描述**: {row['description'][:100] if pd.notna(row['description']) else '暂无'}...\n\n"
        
        return output
        
    except Exception as e:
        return f"❌ 错误: {str(e)}"

def analyze_intent(query):
    """分析用户意图"""
    try:
        from llm4rec.intent_understanding import IntentUnderstandingModule
        
        module = IntentUnderstandingModule(use_template=True)
        intent = module.understand(query)
        
        output = f"## 意图分析结果\n\n"
        output += f"**原始查询**: {query}\n\n"
        output += f"### 提取信息\n\n"
        output += f"- 🗺️ **目标省份**: {intent.get('province') or '未识别'}\n"
        output += f"- 📅 **期望天数**: {intent.get('duration_days') or '未指定'}天\n"
        output += f"- 🎯 **兴趣点**: {', '.join(intent.get('interests', ['未识别']))}\n"
        output += f"- 🎬 **活动类型**: {', '.join(intent.get('activities', ['未识别']))}\n"
        output += f"- 🌸 **季节偏好**: {intent.get('season_preference') or '未指定'}\n"
        output += f"- 🎨 **旅行风格**: {intent.get('travel_style', '观光游')}\n"
        
        if intent.get('constraints'):
            output += f"- ⚠️ **约束条件**: {', '.join(intent['constraints'])}\n"
        
        return output
        
    except Exception as e:
        return f"❌ 错误: {str(e)}"

def show_stats():
    """显示系统统计"""
    try:
        if os.path.exists('data/poi.csv'):
            df = pd.read_csv('data/poi.csv')
            
            output = f"## 📊 数据统计\n\n"
            output += f"- **景点总数**: {len(df)}个\n"
            output += f"- **省份数**: {df['province'].nunique()}个\n\n"
            
            output += f"### 省份分布\n\n"
            prov_counts = df['province'].value_counts()
            for prov, count in prov_counts.items():
                pct = count / len(df) * 100
                bar = '█' * int(pct / 2)
                output += f"- **{prov}**: {count}个 ({pct:.1f}%) {bar}\n"
            
            # 检查向量状态
            vector_status = "✅ 已生成" if os.path.exists('outputs/emb/poi_emb.npy') else "❌ 未生成"
            output += f"\n### 系统状态\n\n"
            output += f"- **POI向量**: {vector_status}\n"
            
            if os.path.exists('data/user_events.csv'):
                events = pd.read_csv('data/user_events.csv')
                output += f"- **用户事件**: {len(events)}条\n"
            
            return output
        else:
            return "❌ 数据文件不存在"
            
    except Exception as e:
        return f"❌ 错误: {str(e)}"

# 创建UI
with gr.Blocks(title="GoAfar 智能旅行推荐", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🎒 GoAfar 智能旅行路线推荐系统
    
    > 基于 **BGE-M3** / **RecBole** / **OR-Tools** / **Qwen3** 的AI推荐系统
    
    **性能**: GPU加速600倍 | 召回率+30% | 可行率92% | 覆盖8省份1333景点
    """)
    
    with gr.Tabs():
        # Tab 1: 语义检索
        with gr.Tab("🔍 语义检索"):
            gr.Markdown("### 输入查询，秒级返回相关景点")
            
            with gr.Row():
                with gr.Column(scale=2):
                    search_input = gr.Textbox(
                        label="搜索查询",
                        placeholder="试试输入：雪山、草原、古城、寺庙、湖泊...",
                        lines=2
                    )
                    search_topk = gr.Slider(5, 20, value=10, step=1, label="返回数量")
                    search_btn = gr.Button("🔍 搜索", variant="primary")
                
                with gr.Column(scale=1):
                    gr.Markdown("""
                    **示例查询**:
                    - 想去看雪山和湖泊
                    - 新疆的草原
                    - 西藏的寺庙和圣湖
                    - 云南的古城
                    - 秋天的胡杨林
                    """)
            
            search_output = gr.Markdown()
            
            search_btn.click(
                fn=test_semantic_search,
                inputs=[search_input, search_topk],
                outputs=search_output
            )
            
            # 示例
            gr.Examples(
                examples=[
                    ["想去看雪山和湖泊", 10],
                    ["新疆的草原", 8],
                    ["西藏的寺庙", 10],
                    ["云南古城", 10],
                ],
                inputs=[search_input, search_topk]
            )
        
        # Tab 2: 意图理解
        with gr.Tab("🤖 意图理解"):
            gr.Markdown("### 测试AI对旅游需求的理解能力")
            
            with gr.Row():
                with gr.Column(scale=2):
                    intent_input = gr.Textbox(
                        label="旅游需求",
                        placeholder="例如：想去新疆喀纳斯看3天秋天的景色，拍照",
                        lines=3
                    )
                    intent_btn = gr.Button("🧠 分析意图", variant="primary")
                
                with gr.Column(scale=1):
                    gr.Markdown("""
                    **AI会提取**:
                    - 目标省份
                    - 期望天数
                    - 兴趣点
                    - 活动类型
                    - 季节偏好
                    - 旅行风格
                    """)
            
            intent_output = gr.Markdown()
            
            intent_btn.click(
                fn=analyze_intent,
                inputs=intent_input,
                outputs=intent_output
            )
            
            gr.Examples(
                examples=[
                    "想去新疆喀纳斯看3天秋天的景色，拍照",
                    "西藏拉萨5日深度游，布达拉宫和纳木错",
                    "云南大理洱海2天骑行，轻松休闲",
                    "四川九寨沟黄龙，亲子游不要太累",
                ],
                inputs=intent_input
            )
        
        # Tab 3: 系统信息
        with gr.Tab("ℹ️ 系统信息"):
            stats_display = gr.Markdown(show_stats())
            refresh_btn = gr.Button("🔄 刷新统计")
            refresh_btn.click(fn=show_stats, outputs=stats_display)
            
            gr.Markdown("""
            ---
            
            ### 📖 项目信息
            
            - **GitHub**: https://github.com/yiweinanzi/trave_afar
            - **完整文档**: 项目完整文档.md
            - **简历材料**: outputs/简历-项目描述.md
            
            ### 🔧 核心技术
            
            - **BGE-M3**: 语义检索（669.7 POI/秒）
            - **RecBole**: 序列推荐（SASRec）
            - **OR-Tools**: VRPTW路线规划
            - **Qwen3-8B**: LLM增强（可选）
            
            ### 📞 联系方式
            
            - **Email**: 2268867257@qq.com
            - **作者**: yiweinanzi
            
            ---
            
            **更新**: 2025-11-08 | **状态**: ✅ Production Ready
            """)

if __name__ == "__main__":
    print("="*80)
    print("GoAfar Web UI 启动中...")
    print("="*80)
    
    # 检查数据文件
    if not os.path.exists('data/poi.csv'):
        print("❌ 错误: data/poi.csv 不存在")
        print("请先运行数据提取脚本")
        exit(1)
    
    print("\n✓ 数据文件检查通过")
    
    if not os.path.exists('outputs/emb/poi_emb.npy'):
        print("⚠️ 警告: POI向量未生成")
        print("   部分功能可能不可用")
        print("   建议运行: python src/embedding/build_embeddings_gpu.py")
    else:
        print("✓ POI向量文件存在")
    
    print("\n正在启动Gradio...")
    print("="*80)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # 生成公网链接
        show_error=True
    )

