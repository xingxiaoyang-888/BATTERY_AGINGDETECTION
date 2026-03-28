# views/components.py
import streamlit as st
from config.settings import THEME_COLOR

def render_kpi_card(title, value, sub, color):
    """渲染通用的 KPI 指标卡片"""
    return f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value" style="color: {color};">{value}</div>
        <div class="kpi-sub" style="color: {color}; opacity: 0.8;">{sub}</div>
    </div>
    """

def render_ai_insight_box(title, insights_list):
    """渲染 AI 智能诊断建议框 (科幻风格)"""
    if not insights_list:
        return ""
        
    content_html = "<br>".join(insights_list)
    return f"""
    <div style="
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-left: 4px solid {THEME_COLOR};
        border-radius: 5px;
        padding: 15px;
        margin-top: 15px;
        color: #ffffff;
        font-size: 14px;
        line-height: 1.6;">
        <div style="font-weight: bold; font-size: 16px; color: {THEME_COLOR}; margin-bottom: 8px;">
            {title}
        </div>
        {content_html}
    </div>
    """