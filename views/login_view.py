# views/login_view.py
import streamlit as st
import time
from utils.db_manager import verify_user, register_user

def render():
    # 保持您的布局逻辑
    col1, col2, col3 = st.columns([1, 1.2, 1])
    
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; color: #00f2ff;'>BATTERY TWIN SYSTEM</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #64748b; letter-spacing: 2px;'>动力电池全生命周期数字孪生平台 V2.0</p>", unsafe_allow_html=True)
        st.markdown("---")
        
        # [新增] 使用 Tabs 区分登录和注册
        tab_login, tab_reg = st.tabs(["🚀 登录 (Login)", "📝 注册 (Register)"])
        
        with tab_login:
            with st.container(border=True):
                username = st.text_input("👤 操作员 ID / Username", placeholder="admin")
                password = st.text_input("🔑 访问密钥 / Password", type="password", placeholder="1234")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                if st.button("🚀 安全连接 (SECURE LOGIN)", use_container_width=True):
                    # [修改] 对接数据库验证
                    if verify_user(username, password):
                        with st.spinner("正在验证身份凭证..."):
                            time.sleep(0.5)
                            st.session_state['logged_in'] = True
                            st.session_state['username'] = username
                            st.success("验证通过！正在加载数字孪生引擎...")
                            time.sleep(0.5)
                            st.rerun()
                    else:
                        st.error("❌ 访问拒绝：账号或密码错误")

        with tab_reg:
            with st.container(border=True):
                new_user = st.text_input("设置新 ID", key="reg_u")
                new_pass = st.text_input("设置新密码", type="password", key="reg_p")
                if st.button("提交注册 (Register)", use_container_width=True):
                    if register_user(new_user, new_pass):
                        st.success(f"用户 {new_user} 注册成功！请切换到登录页登录。")
                    else:
                        st.error("注册失败：用户名可能已存在")

        st.markdown("<p style='text-align: center; color: #334155; font-size: 12px; margin-top: 50px;'>Powered by OpenModelica & Python | Industrial Edition</p>", unsafe_allow_html=True)