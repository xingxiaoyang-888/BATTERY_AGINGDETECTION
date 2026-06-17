# views/components.py
import streamlit as st
from config.settings import THEME_COLOR

# ── 工业级暗色主题配色 ──
CARD_BG = 'rgba(18, 25, 38, 0.85)'
CARD_BORDER = 'rgba(0, 242, 255, 0.15)'
GLOW_COLOR = 'rgba(0, 242, 255, 0.08)'


def render_kpi_card(title, value, sub, color):
    """工业级玻璃拟态 KPI 卡片"""
    return f"""
    <div style="
        background: linear-gradient(135deg, {CARD_BG} 0%, rgba(12, 16, 24, 0.9) 100%);
        border: 1px solid {CARD_BORDER};
        border-radius: 8px;
        padding: 18px 14px;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 2px 12px {GLOW_COLOR};
    ">
        <div style="
            position: absolute; top: 0; left: 0; width: 100%; height: 2px;
            background: linear-gradient(90deg, transparent, {color}, transparent);
        "></div>
        <div style="
            font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px;
            color: #8892a4; margin-bottom: 6px; font-weight: 500;
        ">{title}</div>
        <div style="
            font-size: 28px; font-weight: 700; color: {color};
            font-family: 'Consolas', 'Courier New', monospace;
            letter-spacing: -0.5px; margin-bottom: 4px;
        ">{value}</div>
        <div style="
            font-size: 12px; color: #5a6780;
        ">{sub}</div>
    </div>
    """


def render_ai_insight_box(title, insights_list):
    """工业级诊断面板"""
    if not insights_list:
        return ""
    items = "".join(
        f'<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.04);">{item}</div>'
        for item in insights_list
    )
    return f"""
    <div style="
        background: linear-gradient(135deg, rgba(18,25,38,0.9), rgba(12,16,24,0.95));
        border: 1px solid rgba(0,242,255,0.12);
        border-radius: 8px;
        padding: 16px 18px;
        margin-top: 12px;
        color: #c8d6e5;
        font-size: 13px;
        line-height: 1.7;
        box-shadow: 0 2px 16px rgba(0,0,0,0.3);
    ">
        <div style="
            font-weight: 600; font-size: 14px; color: {THEME_COLOR};
            margin-bottom: 8px; letter-spacing: 0.5px;
        "> {title}</div>
        {items}
    </div>
    """
