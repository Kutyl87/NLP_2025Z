import os
import re
import base64
import markdown


def embed_images_in_markdown(md_path: str) -> None:
    """Embed images in markdown file as base64 data URIs in HTML img tags."""
    project_root = os.getcwd()

    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    def replace_img_tag(match):
        full_tag = match.group(0)
        src_match = re.search(r'src=["\']([^"\']+)["\']', full_tag)
        if not src_match:
            return full_tag

        img_path = src_match.group(1)

        if img_path.startswith(('data:', 'file://', 'http://', 'https://')):
            return full_tag

        if not os.path.isabs(img_path):
            abs_img_path = os.path.abspath(os.path.join(project_root, img_path))
        else:
            abs_img_path = os.path.abspath(img_path)

        if os.path.exists(abs_img_path):
            try:
                with open(abs_img_path, 'rb') as img_file:
                    img_data = img_file.read()
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                    ext = os.path.splitext(abs_img_path)[1].lower()
                    mime_types = {
                        '.png': 'image/png',
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.gif': 'image/gif',
                        '.svg': 'image/svg+xml',
                        '.webp': 'image/webp'
                    }
                    mime_type = mime_types.get(ext, 'image/png')
                    data_uri = f'data:{mime_type};base64,{img_base64}'
                    return full_tag.replace(f'src="{img_path}"', f'src="{data_uri}"').replace(f"src='{img_path}'", f"src='{data_uri}'")
            except Exception:
                return full_tag

        return full_tag

    img_pattern = r'<img[^>]+>'
    md_content = re.sub(img_pattern, replace_img_tag, md_content)

    def replace_markdown_image(match):
        alt_text = match.group(1)
        img_path = match.group(2)

        if img_path.startswith(('data:', 'file://', 'http://', 'https://')):
            return match.group(0)

        if not os.path.isabs(img_path):
            abs_img_path = os.path.abspath(os.path.join(project_root, img_path))
        else:
            abs_img_path = os.path.abspath(img_path)

        if os.path.exists(abs_img_path):
            try:
                with open(abs_img_path, 'rb') as img_file:
                    img_data = img_file.read()
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                    ext = os.path.splitext(abs_img_path)[1].lower()
                    mime_types = {
                        '.png': 'image/png',
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.gif': 'image/gif',
                        '.svg': 'image/svg+xml',
                        '.webp': 'image/webp'
                    }
                    mime_type = mime_types.get(ext, 'image/png')
                    data_uri = f'data:{mime_type};base64,{img_base64}'
                    return f'<img src="{data_uri}" alt="{alt_text}" />'
            except Exception:
                return match.group(0)

        return match.group(0)

    md_image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    md_content = re.sub(md_image_pattern, replace_markdown_image, md_content)

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)


def convert_markdown_to_html(md_path: str, html_path: str) -> None:
    """Convert markdown file to HTML with embedded images as base64 data URIs for standalone viewing."""
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    html_content = markdown.markdown(
        md_content,
        extensions=['extra', 'codehilite', 'tables']
    )

    project_root = os.getcwd()
    html_dir = os.path.dirname(os.path.abspath(html_path))

    def fix_img_tag(match):
        full_tag = match.group(0)
        src_match = re.search(r'src=["\']([^"\']+)["\']', full_tag)
        if not src_match:
            return full_tag

        img_path = src_match.group(1)

        if img_path.startswith(('data:', 'file://', 'http://', 'https://')):
            return full_tag


        if not os.path.isabs(img_path):
            abs_img_path = os.path.abspath(os.path.join(project_root, img_path))
        else:
            abs_img_path = os.path.abspath(img_path)

        if os.path.exists(abs_img_path):
            try:
                with open(abs_img_path, 'rb') as img_file:
                    img_data = img_file.read()
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                    ext = os.path.splitext(abs_img_path)[1].lower()
                    mime_types = {
                        '.png': 'image/png',
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.gif': 'image/gif',
                        '.svg': 'image/svg+xml',
                        '.webp': 'image/webp'
                    }
                    mime_type = mime_types.get(ext, 'image/png')
                    data_uri = f'data:{mime_type};base64,{img_base64}'
                    return full_tag.replace(f'src="{img_path}"', f'src="{data_uri}"').replace(f"src='{img_path}'", f"src='{data_uri}'")
            except Exception:
                try:
                    rel_path = os.path.relpath(abs_img_path, html_dir)
                    rel_path = rel_path.replace('\\', '/')
                    return full_tag.replace(f'src="{img_path}"', f'src="{rel_path}"').replace(f"src='{img_path}'", f"src='{rel_path}'")
                except ValueError:
                    return full_tag

        return full_tag

    img_pattern = r'<img[^>]+>'
    html_content = re.sub(img_pattern, fix_img_tag, html_content)

    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Analysis Report</title>
    <style>
        body {{
            font-family: 'DejaVu Sans', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
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
            border: 1px solid #ddd;
            border-radius: 4px;
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
            border: 1px solid #ddd;
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
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
    </style>
</head>
<body>
    <div class="container">
        {html_content}
    </div>
</body>
</html>"""

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(full_html)

