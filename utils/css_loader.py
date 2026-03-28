# utils/css_loader.py
import streamlit as st
import os

def load_css(file_name="assets/styles.css"):
    """
    读取本地 CSS 文件并注入到 Streamlit 页面
    """
    try:
        # [关键修复] 增加 encoding="utf-8" 参数，防止 Windows 下报 GBK 解码错误
        with open(file_name, "r", encoding="utf-8") as f:
            css_content = f.read()
            st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"⚠️ 样式文件未找到: {file_name}")
    except Exception as e:
        st.error(f"⚠️ 样式加载失败: {e}")