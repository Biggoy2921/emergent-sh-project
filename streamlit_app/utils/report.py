#!/usr/bin/env python3
"""
PDF Report Generator
Generate detailed analysis reports with visualizations
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
import os

class MalwareReport:
    """Generate PDF reports for malware analysis"""
    
    def __init__(self, filename):
        self.filename = filename
        self.doc = SimpleDocTemplate(filename, pagesize=letter)
        self.styles = getSampleStyleSheet()
        self.story = []
        
        # Custom styles
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#ff0066'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#00ff00'),
            spaceAfter=12
        )
    
    def add_title(self, title):
        """Add report title"""
        self.story.append(Paragraph(title, self.title_style))
        self.story.append(Spacer(1, 0.2*inch))
    
    def add_section(self, heading, content):
        """Add a section with heading and content"""
        self.story.append(Paragraph(heading, self.heading_style))
        self.story.append(Paragraph(content, self.styles['Normal']))
        self.story.append(Spacer(1, 0.2*inch))
    
    def add_table(self, data, col_widths=None):
        """Add a styled table"""
        table = Table(data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        self.story.append(table)
        self.story.append(Spacer(1, 0.3*inch))
    
    def add_image(self, image_path, width=6*inch):
        """Add an image to the report"""
        if os.path.exists(image_path):
            img = Image(image_path, width=width)
            self.story.append(img)
            self.story.append(Spacer(1, 0.2*inch))
    
    def generate_malware_report(self, analysis_results):
        """Generate complete malware analysis report"""
        
        # Title
        self.add_title("MALWARE ANALYSIS REPORT")
        
        # Report metadata
        metadata = f"""
        <b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        <b>File Name:</b> {analysis_results.get('filename', 'Unknown')}<br/>
        <b>File Size:</b> {analysis_results.get('file_size', 'Unknown')} bytes<br/>
        <b>File Type:</b> {analysis_results.get('file_type', 'Unknown')}
        """
        self.add_section("📄 File Information", metadata)
        
        # Detection results
        prediction = analysis_results.get('prediction', 0)
        probability = analysis_results.get('probability', 0)
        threat_level = analysis_results.get('threat_level', 'UNKNOWN')
        malware_family = analysis_results.get('malware_family', 'Unknown')
        
        detection_text = f"""
        <b>Status:</b> {'<font color="red">MALICIOUS</font>' if prediction == 1 else '<font color="green">CLEAN</font>'}<br/>
        <b>Confidence:</b> {probability*100:.2f}%<br/>
        <b>Threat Level:</b> {threat_level}<br/>
        <b>Classification:</b> {malware_family}
        """
        self.add_section("⚠️ Detection Results", detection_text)
        
        # Model predictions table
        individual_probs = analysis_results.get('individual_probabilities', {})
        table_data = [['Model', 'Prediction', 'Confidence']]
        
        for model, prob in individual_probs.items():
            pred = 'Malware' if prob > 0.5 else 'Clean'
            table_data.append([model, pred, f'{prob*100:.2f}%'])
        
        if len(table_data) > 1:
            self.add_section("🤖 Model Predictions", "")
            self.add_table(table_data, col_widths=[2*inch, 2*inch, 2*inch])
        
        # Key features (if available)
        features = analysis_results.get('features', {})
        if features:
            features_text = "<br/>".join([f"<b>{k}:</b> {v}" for k, v in list(features.items())[:10]])
            self.add_section("🔍 Key Features", features_text)
        
        # Recommendations
        if prediction == 1:
            recommendations = """
            • Do NOT execute this file<br/>
            • Quarantine or delete immediately<br/>
            • Run full system scan<br/>
            • Check system for signs of compromise<br/>
            • Update antivirus definitions
            """
        else:
            recommendations = """
            • File appears to be clean<br/>
            • Verify file source before execution<br/>
            • Keep antivirus software updated<br/>
            • Monitor for suspicious behavior
            """
        
        self.add_section("🛡️ Recommendations", recommendations)
        
        # Disclaimer
        disclaimer = """
        <i>This report is generated by an automated machine learning system. 
        While highly accurate (99% for Decision Tree), no detection system is perfect. 
        Always exercise caution with unknown files.</i>
        """
        self.add_section("⚠️ Disclaimer", disclaimer)
    
    def build(self):
        """Build and save the PDF report"""
        self.doc.build(self.story)
        return self.filename

def generate_report(analysis_results, output_path):
    """Convenience function to generate report"""
    report = MalwareReport(output_path)
    report.generate_malware_report(analysis_results)
    report.build()
    return output_path
