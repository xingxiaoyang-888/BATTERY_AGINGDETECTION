# main.py
import streamlit as st
import time

# 必须是第一个 Streamlit 命令
st.set_page_config(
    page_title="Battery Twin Pro",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 导入模块 ---
from config.settings import apply_custom_css
from views.dashboard_view import render_dashboard
# [新增] 导入数据库工具
from utils.db_manager import init_db, verify_user, register_user

# --- 初始化 ---
# 1. 初始化数据库 (如果不存在则创建 battery_app.db)
init_db()

# 2. 注入 CSS (应用样式)
apply_custom_css()

# 3. Session 状态管理
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'auth_mode' not in st.session_state:
    st.session_state['auth_mode'] = 'login' # 控制显示登录还是注册

# --- 路由逻辑 ---
def login_page():
    """独立的登录页面视图 (融合数据库验证)"""
    # 调整列宽，让中间宽一点
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        
        #  标题区 (保留您喜欢的蓝色标题)
        st.markdown("""
        <div style="text-align:center;margin:20px 0 30px 0;">
            <h1 style="font-size:30px;font-weight:700;color:#00f2ff;margin:0;letter-spacing:2px;">
            Sodium-Ion Battery Digital Twin
            </h1>
            <p style="color:#5a6a80;font-size:13px;margin-top:6px;">
            Industrial Electro-Thermal-Aging Simulation Platform
            </p>
            <p style="color:#3a4a60;font-size:11px;">
            Powered by OpenModelica FMI 2.0 · Multi-FMU Architecture
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        
        # 使用 Tabs 切换登录/注册 (这是最优雅的方式，不破坏您的布局)
        tab_login, tab_reg = st.tabs([" 登录系统 (Login)", " 注册新账户 (Register)"])
        
        # === 登录 Tab ===
        with tab_login:
            with st.form("login_form"):
                u = st.text_input("Operator ID", placeholder="admin")
                p = st.text_input("Password", type="password", placeholder="1234")
                
                # 按钮样式已经在 settings.py 中被强制改为黑色文字了
                submitted = st.form_submit_button("LOGIN / 登录", use_container_width=True)
                
                if submitted:
                    # [修改点] 调用数据库验证
                    if verify_user(u, p):
                        st.success(" 身份验证通过")
                        time.sleep(0.5)
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = u # 记录用户名用于显示头像
                        st.rerun()
                    else:
                        st.error(" 账号或密码错误 (请重试或注册)")

        # === 注册 Tab (新增) ===
        with tab_reg:
            with st.form("reg_form"):
                new_u = st.text_input("设置新 ID (New Username)")
                new_p = st.text_input("设置新密码 (New Password)", type="password")
                
                reg_submitted = st.form_submit_button("REGISTER / 注册", use_container_width=True)
                
                if reg_submitted:
                    if new_u and new_p:
                        # [修改点] 调用数据库注册
                        success, msg = register_user(new_u, new_p)
                        if success:
                            st.success(f" 用户 {new_u} 注册成功！请切换到登录页登录。")
                        else:
                            st.error(f" {msg}")
                    else:
                        st.warning(" 请填写完整信息")

        st.markdown("<br><p style='text-align: center; color: #555; font-size: 12px;'>Powered by OpenModelica & Python | Industrial Edition</p>", unsafe_allow_html=True)

# --- 主程序流 ---
if not st.session_state['logged_in']:
    login_page()
else:
    render_dashboard()