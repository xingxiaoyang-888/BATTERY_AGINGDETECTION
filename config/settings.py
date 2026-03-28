# config/settings.py
from utils.css_loader import load_css

# === 核心配色 ===
THEME_COLOR = "#00f2ff"
SUCCESS_COLOR = "#00e676"
WARNING_COLOR = "#ffab00"
DANGER_COLOR = "#ff1744"
NEUTRAL_COLOR = "#9e9e9e"

def apply_custom_css():
    """加载外部 CSS 文件"""
    load_css("assets/styles.css")