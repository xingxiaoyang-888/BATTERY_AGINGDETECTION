"""前端通用工业化组件。"""

from html import escape


def render_kpi_card(title, value, sub, color="#45d2be", tag="LIVE"):
    return f"""
    <div class="kpi-card" style="--accent:{escape(str(color))}">
        <div class="kpi-title">{escape(str(title))}</div>
        <div class="kpi-value">{escape(str(value))}</div>
        <div class="kpi-sub">{escape(str(sub))}</div>
        <span style="position:absolute;right:12px;top:11px;color:{escape(str(color))};
                     font-size:9px;letter-spacing:.12em;opacity:.75;">{escape(str(tag))}</span>
    </div>
    """


def render_section_header(eyebrow, title, description=""):
    return f"""
    <div class="panel-heading">
        <div class="section-eyebrow">{escape(str(eyebrow))}</div>
        <h3>{escape(str(title))}</h3>
        <p>{escape(str(description))}</p>
    </div>
    """


def render_empty_state(code, title, description, action=""):
    action_html = f'<div class="empty-action">{escape(action)}</div>' if action else ""
    return f"""
    <div class="empty-state">
        <div class="empty-code">{escape(str(code))}</div>
        <h3>{escape(str(title))}</h3>
        <p>{escape(str(description))}</p>
        {action_html}
    </div>
    """


def render_ai_insight_box(title, insights_list):
    if not insights_list:
        return ""
    items = "".join(
        f'<div class="insight-item">{item}</div>' for item in insights_list
    )
    return f"""
    <div class="insight-panel">
        <div class="section-eyebrow">INTELLIGENCE LAYER</div>
        <h4>{escape(str(title))}</h4>
        {items}
    </div>
    """
