# utils/report_generator.py
from fpdf import FPDF
import tempfile
import os

class BatteryReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Battery Digital Twin - Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def safe_text(text):
    """
    [关键修复] 清洗文本，移除无法编码的中文字符
    FPDF 标准版不支持中文，为了防止崩溃，我们将中文替换为 '?'
    """
    try:
        # 尝试编码为 latin-1，失败则替换
        return str(text).encode('latin-1', 'replace').decode('latin-1')
    except Exception:
        return "Unknown Text"

def generate_pdf_report(username, config, kpis, df_summary):
    """
    生成 PDF 报告 (增强健壮性)
    """
    try:
        pdf = BatteryReport()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # 1. 基本信息
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"Operator: {safe_text(username)}", ln=True)
        pdf.cell(0, 10, "-"*50, ln=True)
        
        # 2. 仿真配置
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "1. Configuration", ln=True)
        pdf.set_font("Arial", size=11)
        
        for key, val in config.items():
            if isinstance(val, (str, int, float)) and key not in ['uploaded_file']:
                # 使用 safe_text 过滤中文
                pdf.cell(0, 8, f"{safe_text(key)}: {safe_text(val)}", ln=True)
                
        # 3. 核心 KPI
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "2. Key Indicators", ln=True)
        pdf.set_font("Arial", size=11)
        
        pdf.cell(0, 8, f"SOH: {kpis.get('soh', 0):.4f} %", ln=True)
        pdf.cell(0, 8, f"Max Temp: {kpis.get('max_temp', 0):.1f} C", ln=True)
        
        # 4. 统计摘要
        if df_summary is not None and not df_summary.empty:
            pdf.ln(10)
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "3. Data Summary", ln=True)
            pdf.set_font("Arial", size=11)
            duration = df_summary['Time'].iloc[-1]
            pdf.cell(0, 8, f"Duration: {duration:.1f} s", ln=True)

        # 保存
        tmp_file = tempfile.mktemp(suffix=".pdf")
        pdf.output(tmp_file)
        
        with open(tmp_file, "rb") as f:
            pdf_bytes = f.read()
        
        os.remove(tmp_file)
        return pdf_bytes

    except Exception as e:
        print(f"🔴 PDF生成失败: {e}") # 打印错误到控制台
        return None