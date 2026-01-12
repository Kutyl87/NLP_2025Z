from __future__ import annotations
import os
import re
from pathlib import Path
from io import BytesIO
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
from core.orchestrator import Orchestrator
from utils.utils import ensure_dirs
import markdown
from weasyprint import HTML

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'data/input'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Ensure directories exist
ensure_dirs()
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def convert_markdown_to_pdf(md_path: str, pdf_path: str) -> None:
    """Convert markdown file to PDF with embedded images."""
    # Read markdown file
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Convert markdown to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=['extra', 'codehilite', 'tables']
    )

    # Fix image paths in HTML - convert relative paths to absolute file:// URLs
    # Plot paths are relative to project root, not markdown file directory
    project_root = os.getcwd()

    # Find all img tags and fix their src attributes
    def fix_img_tag(match):
        full_tag = match.group(0)
        # Extract src attribute value
        src_match = re.search(r'src=["\']([^"\']+)["\']', full_tag)
        if not src_match:
            return full_tag

        img_path = src_match.group(1)

        # Skip if already a file:// URL or http(s):// URL
        if img_path.startswith(('file://', 'http://', 'https://')):
            return full_tag

        # If path is relative, resolve it relative to project root
        if not os.path.isabs(img_path):
            # Try relative to project root first (plots are stored this way)
            abs_img_path = os.path.abspath(os.path.join(project_root, img_path))
            # If that doesn't exist, try relative to markdown file directory
            if not os.path.exists(abs_img_path):
                md_dir = os.path.dirname(os.path.abspath(md_path))
                abs_img_path = os.path.abspath(os.path.join(md_dir, img_path))
        else:
            abs_img_path = os.path.abspath(img_path)

        # Check if file exists
        if os.path.exists(abs_img_path):
            # Replace src with file:// URL
            new_src = f'file://{abs_img_path}'
            return full_tag.replace(f'src="{img_path}"', f'src="{new_src}"').replace(f"src='{img_path}'", f"src='{new_src}'")

        # If file doesn't exist, return original
        return full_tag

    # Replace img tags with fixed src attributes
    img_pattern = r'<img[^>]+>'
    html_content = re.sub(img_pattern, fix_img_tag, html_content)

    # Wrap in proper HTML structure
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @page {{
                size: A4;
                margin: 2cm;
            }}
            body {{
                font-family: 'DejaVu Sans', Arial, sans-serif;
                line-height: 1.6;
                color: #333;
            }}
            h1 {{
                color: #8b5cf6;
                border-bottom: 2px solid #8b5cf6;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #3b82f6;
                margin-top: 30px;
            }}
            h3 {{
                color: #f97316;
            }}
            img {{
                max-width: 100%;
                height: auto;
                margin: 15px 0;
                display: block;
                page-break-inside: avoid;
            }}
            code {{
                background-color: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }}
            pre {{
                background-color: #f4f4f4;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 15px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #8b5cf6;
                color: white;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Convert HTML to PDF
    # Use explicit string parameter to avoid guessing issues
    html_doc = HTML(string=full_html)
    # Write PDF to BytesIO first, then to file (avoids WeasyPrint API issues)
    pdf_bytes = BytesIO()
    html_doc.write_pdf(pdf_bytes)
    pdf_bytes.seek(0)
    with open(pdf_path, 'wb') as f:
        f.write(pdf_bytes.getvalue())


@app.route('/')
def index():
    """Main page with file upload and prompt input."""
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    """Process the uploaded file and prompt."""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        prompt = request.form.get('prompt', '').strip()

        # Validate file
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Only CSV files are allowed'}), 400

        # Validate prompt
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Initialize orchestrator with the uploaded file
        orch = Orchestrator(data_path=file_path)

        # Run the pipeline
        result = orch.run(input_text=prompt, data_path=file_path)

        # Get report path (markdown)
        report_md_path = result.get('report_path')
        if not report_md_path or not os.path.exists(report_md_path):
            return jsonify({'error': 'Report generation failed'}), 500

        # Convert markdown to PDF
        report_pdf_path = report_md_path.replace('.md', '.pdf')
        pdf_success = False
        try:
            convert_markdown_to_pdf(report_md_path, report_pdf_path)
            pdf_success = os.path.exists(report_pdf_path)
        except Exception as pdf_error:
            # If PDF conversion fails, log but continue
            app.logger.warning(f'PDF conversion failed: {pdf_error}')
            pdf_success = False

        # Return both filenames if available
        report_md_filename = os.path.basename(report_md_path)
        report_pdf_filename = os.path.basename(report_pdf_path) if pdf_success else None

        return jsonify({
            'success': True,
            'report_md_filename': report_md_filename,
            'report_pdf_filename': report_pdf_filename,
            'message': 'Work is done!'
        })

    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500


@app.route('/download/<filename>')
def download_file(filename: str):
    """Download the generated report file."""
    try:
        report_path = os.path.join('data/output', secure_filename(filename))
        if not os.path.exists(report_path):
            return jsonify({'error': 'File not found'}), 404
        return send_file(report_path, as_attachment=True)
    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

