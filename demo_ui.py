"""
GoAfar 简化演示UI
"""
import gradio as gr
import pandas as pd
import os

def search_demo(query):
    """简单的搜索演示"""
    if not os.path.exists('data/poi.csv'):
        return "数据文件不存在"
    
    df = pd.read_csv('data/poi.csv')
    
    # 简单的关键词搜索
    mask = df['name'].str.contains(query, case=False, na=False) | \
           df['description'].str.contains(query, case=False, na=False)
    
    results = df[mask].head(10)
    
    if len(results) == 0:
        return f"未找到包含'{query}'的景点"
    
    output = f"找到 {len(results)} 个相关景点:\n\n"
    for i, (_, row) in enumerate(results.iterrows(), 1):
        output += f"{i}. **{row['name']}** - {row['city']}, {row['province']}\n"
    
    return output

# 创建UI
demo = gr.Interface(
    fn=search_demo,
    inputs=gr.Textbox(label="搜索景点", placeholder="输入关键词，如：湖、山、草原..."),
    outputs=gr.Markdown(label="搜索结果"),
    title="🎒 GoAfar 智能旅行推荐系统",
    description="基于BGE-M3/RecBole/OR-Tools/Qwen3 | GPU加速600倍 | 1333景点",
    examples=[["湖"], ["草原"], ["古城"], ["雪山"]],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

