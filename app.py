from __future__ import annotations

import os
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, send_file, jsonify

from core.orchestrator import Orchestrator
from utils.utils import ensure_dirs, allowed_file
from utils.app_utils import embed_images_in_markdown, convert_markdown_to_html


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'data/input'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}


ensure_dirs()
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    """Main page with file upload and prompt input."""
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    """Process the uploaded file and prompt."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        prompt = request.form.get('prompt', '').strip()

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
            return jsonify({'error': 'Only CSV files are allowed'}), 400

        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        orch = Orchestrator(data_path=file_path)
        result = orch.run(input_text=prompt, data_path=file_path)

        report_md_path = result.get('report_path')
        if not report_md_path or not os.path.exists(report_md_path):
            return jsonify({'error': 'Report generation failed'}), 500

        try:
            embed_images_in_markdown(report_md_path)
        except Exception as md_error:
            app.logger.warning(f'Failed to embed images in markdown: {md_error}')

        report_html_path = report_md_path.replace('.md', '.html')
        try:
            convert_markdown_to_html(report_md_path, report_html_path)
            html_success = os.path.exists(report_html_path)
        except Exception as html_error:
            app.logger.warning(f'HTML conversion failed: {html_error}')
            html_success = False

        report_md_filename = os.path.basename(report_md_path)
        report_html_filename = os.path.basename(report_html_path) if html_success else None

        return jsonify({
            'success': True,
            'report_md_filename': report_md_filename,
            'report_html_filename': report_html_filename,
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

