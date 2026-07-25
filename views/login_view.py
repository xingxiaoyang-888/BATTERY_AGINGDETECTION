"""平台登录与注册视图。"""

import time

import streamlit as st

from utils.db_manager import register_user, verify_user


def render_login():
    st.markdown(
        """
        <section class="login-hero">
            <div class="brand-lockup">
                <div class="brand-mark">Na</div>
                <div>
                    <div class="brand-kicker">SODIUM TWIN OPERATIONS</div>
                    <h1>钠离子电池全生命周期数字孪生平台</h1>
                </div>
            </div>
            <p class="login-lead">
                面向电芯、模组与电池包的物理仿真、热安全诊断、寿命预测与资产健康管理。
            </p>
            <div class="capability-row">
                <span><i class="status-dot online"></i> FMI 2.0 PHYSICS</span>
                <span><i class="status-dot online"></i> RWTH LIFE AI</span>
                <span><i class="status-dot online"></i> 3D THERMAL TWIN</span>
                <span><i class="status-dot standby"></i> COMPANY CALIBRATION</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    left, form_col = st.columns([1.22, 0.78], gap="large")
    with left:
        st.markdown(
            """
            <div class="login-intro-panel">
                <div class="section-eyebrow">PLATFORM CAPABILITIES</div>
                <h2>从瞬态电热响应，到可解释的寿命风险区间</h2>
                <div class="feature-grid">
                    <div class="feature-item"><b>01</b><span>多规格FMU联合仿真</span><small>电压 / SOC / 热场 / 故障注入</small></div>
                    <div class="feature-item"><b>02</b><span>三维空间数字孪生</span><small>温度、SOH与最差电芯定位</small></div>
                    <div class="feature-item"><b>03</b><span>删失感知寿命预测</span><small>RUL、置信区间与OOD检测</small></div>
                    <div class="feature-item"><b>04</b><span>工程证据与审计</span><small>数据覆盖、模型指标与报告</small></div>
                </div>
                <div class="platform-note">
                    <span>ENGINEERING BUILD</span>
                    <strong>Physical Model × Data Intelligence</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with form_col:
        st.markdown(
            """
            <div class="auth-heading">
                <div class="section-eyebrow">SECURE ACCESS</div>
                <h2>进入运行控制台</h2>
                <p>使用本地工程账户访问仿真与寿命分析工作区。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        login_tab, register_tab = st.tabs(["账户登录", "创建账户"])

        with login_tab:
            with st.form("login_form", border=False):
                username = st.text_input("操作员账户", placeholder="请输入账户名称")
                password = st.text_input("访问密码", type="password", placeholder="请输入访问密码")
                submitted = st.form_submit_button("进入控制台", use_container_width=True)
                if submitted:
                    if verify_user(username.strip(), password):
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = username.strip()
                        st.success("身份验证通过，正在载入工作区。")
                        time.sleep(0.3)
                        st.rerun()
                    else:
                        st.error("账户或密码错误，请重新输入。")

        with register_tab:
            with st.form("register_form", border=False):
                new_user = st.text_input("新账户名称", placeholder="至少3个字符")
                new_password = st.text_input("设置密码", type="password", placeholder="至少4个字符")
                confirmed = st.text_input("确认密码", type="password", placeholder="再次输入密码")
                submitted = st.form_submit_button("创建工程账户", use_container_width=True)
                if submitted:
                    if len(new_user.strip()) < 3 or len(new_password) < 4:
                        st.warning("账户至少3个字符，密码至少4个字符。")
                    elif new_password != confirmed:
                        st.warning("两次输入的密码不一致。")
                    else:
                        success, message = register_user(new_user.strip(), new_password)
                        if success:
                            st.success("账户创建成功，请切换至登录页。")
                        else:
                            st.error(message)

        st.markdown(
            """
            <div class="auth-footer">
                <span><i class="status-dot online"></i> Local secure storage</span>
                <span>Version 3.0 · Engineering Preview</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
