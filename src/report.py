from fpdf import FPDF
import datetime

class EEGReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'NeuroGuard Clinical EEG Analysis Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(data):
    pdf = EEGReport()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)
    
    # Metadata
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    pdf.cell(0, 10, f"File Name: {str(data.get('filename', 'N/A'))}", 0, 1)
    pdf.cell(0, 10, f"Model Used: {str(data.get('model_name', 'N/A'))}", 0, 1)
    pdf.ln(5)
    
    # Statistics
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Inference Statistics', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"Max Seizure Risk: {float(data.get('risk_score', 0))*100:.2f}%", 0, 1)
    pdf.cell(0, 10, f"Total Segments: {int(data.get('total_segments', 0))}", 0, 1)
    pdf.cell(0, 10, f"Detections Found: {int(data.get('detections', 0))}", 0, 1)
    pdf.cell(0, 10, f"Accuracy: {str(data.get('accuracy', 'N/A'))}", 0, 1)
    pdf.ln(5)
    
    # XAI
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Explainable AI (XAI) Details', 0, 1)
    pdf.set_font('Arial', '', 12)
    top_chs = data.get('top_channels', [])
    pdf.multi_cell(0, 10, f"Top Contributing Channels: {', '.join([str(c) for c in top_chs])}")
    pdf.ln(5)
    
    # Clinical Notes
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Clinical Findings', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, str(data.get('clinical_notes', "No significant findings recorded.")))
    
    # Return as bytes
    return pdf.output()
