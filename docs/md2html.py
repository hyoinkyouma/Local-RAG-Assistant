import base64
import os
import re

import markdown

ROOT = r"C:\Users\roman\Desktop\Local-RAG-Assistant"
MD = os.path.join(ROOT, "USER_GUIDE.md")
HTML = os.path.join(ROOT, "docs", "USER_GUIDE.html")

with open(MD, encoding="utf-8") as f:
    text = f.read()


def inline_images(m):
    alt, rel = m.group(1), m.group(2)
    full = os.path.join(ROOT, rel)
    if os.path.exists(full):
        b64 = base64.b64encode(open(full, "rb").read()).decode()
        ext = os.path.splitext(full)[1].lstrip(".").lower() or "png"
        return f"![{alt}](data:image/{ext};base64,{b64})"
    return m.group(0)


text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", inline_images, text)

body = markdown.markdown(text, extensions=["tables", "sane_lists", "toc"])

CSS = """
@page { size: A4; margin: 17mm 16mm 19mm 16mm; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  font-family: 'Segoe UI', -apple-system, 'Helvetica Neue', Arial, sans-serif;
  color: #1f2937; font-size: 10.5pt; line-height: 1.55; margin: 0;
}
h1 { color: #1d4ed8; font-size: 20pt; border-bottom: 2px solid #93c5fd; padding-bottom: 8px; margin: 0 0 8px; }
h2 { color: #1e40af; font-size: 14pt; margin: 24px 0 8px; border-bottom: 1px solid #dbeafe; padding-bottom: 4px; }
h3 { font-size: 11.5pt; margin: 16px 0 6px; color: #111827; }
p { margin: 8px 0; }
ul, ol { margin: 8px 0; padding-left: 22px; }
li { margin: 4px 0; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; page-break-inside: avoid; }
th, td { border: 1px solid #9ca3af; padding: 6px 10px; text-align: left; vertical-align: top; font-size: 9.5pt; }
th { background: #eff6ff; color: #1e3a8a; }
tr:nth-child(even) td { background: #f9fafb; }
img { max-width: 100%; border: 1px solid #d1d5db; border-radius: 6px; margin: 10px 0; page-break-inside: avoid; }
blockquote {
  background: #f3f4f6; border-left: 4px solid #60a5fa;
  padding: 8px 14px; margin: 12px 0; border-radius: 4px; color: #374151;
}
code { background: #f3f4f6; padding: 1px 5px; border-radius: 3px; font-size: 0.92em; }
strong { font-weight: 600; }
hr { border: none; border-top: 1px solid #e5e7eb; margin: 20px 0; }
a { color: #2563eb; text-decoration: none; }
"""

html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>DocuStore Local Assistant - User Guide</title>
<style>{CSS}</style></head><body>{body}</body></html>
"""

with open(HTML, "w", encoding="utf-8") as f:
    f.write(html_doc)
print("HTML written:", HTML)
