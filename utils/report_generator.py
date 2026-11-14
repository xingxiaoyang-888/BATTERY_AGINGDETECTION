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
    """Replace characters unsupported by the legacy Latin-1 report font."""
    try:

        return str(text).encode('latin-1', 'replace').decode('latin-1')
    except Exception:
        return "Unknown Text"

def generate_pdf_report(username, config, kpis, df_summary):
    """Generate an engineering PDF using the legacy Latin-1 font contract."""
    try:
        pdf = BatteryReport()
        pdf.add_page()
        pdf.set_font("Arial", size=12)


        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"Operator: {safe_text(username)}", ln=True)
        pdf.cell(0, 10, "-"*50, ln=True)


        pdf.ln(5)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "1. Configuration", ln=True)
        pdf.set_font("Arial", size=11)

        for key, val in config.items():
            if isinstance(val, (str, int, float)) and key not in ['uploaded_file']:

                pdf.cell(0, 8, f"{safe_text(key)}: {safe_text(val)}", ln=True)


        pdf.ln(10)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "2. Key Indicators", ln=True)
        pdf.set_font("Arial", size=11)

        pdf.cell(0, 8, f"SOH: {kpis.get('soh', 0):.4f} %", ln=True)
        pdf.cell(0, 8, f"Max Temp: {kpis.get('max_temp', 0):.1f} C", ln=True)


        if df_summary is not None and not df_summary.empty:
            pdf.ln(10)
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "3. Data Summary", ln=True)
            pdf.set_font("Arial", size=11)
            duration = df_summary['Time'].iloc[-1]
            pdf.cell(0, 8, f"Duration: {duration:.1f} s", ln=True)


        tmp_file = tempfile.mktemp(suffix=".pdf")
        pdf.output(tmp_file)

        with open(tmp_file, "rb") as f:
            pdf_bytes = f.read()

        os.remove(tmp_file)
        return pdf_bytes

    except Exception as e:
        print(f"🔴 PDF生成失败: {e}")
        return None
