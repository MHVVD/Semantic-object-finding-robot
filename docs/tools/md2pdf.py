#!/usr/bin/env python3
# Copyright 2026 Mahmud
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert a milestone report from Markdown to PDF (Python-Markdown + WeasyPrint).

Setup once (no sudo needed):
    python3 -m venv .venv-docs
    .venv-docs/bin/pip install markdown weasyprint

Usage:
    .venv-docs/bin/python docs/tools/md2pdf.py docs/learning/M00_workspace_setup.md
"""

import pathlib
import sys

import markdown
from weasyprint import HTML

CSS = """
@page { size: A4; margin: 18mm 16mm; @bottom-right { content: counter(page); font-size: 9pt; } }
body { font-family: 'DejaVu Sans', sans-serif; font-size: 10.5pt; line-height: 1.45; }
h1 { font-size: 20pt; border-bottom: 2px solid #333; padding-bottom: 4px; }
h2 { font-size: 15pt; margin-top: 22px; border-bottom: 1px solid #aaa; page-break-after: avoid; }
h3 { font-size: 12pt; margin-top: 16px; page-break-after: avoid; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 9pt; background: #f2f2f2; }
pre { background: #f6f6f6; border: 1px solid #ddd; padding: 8px; font-size: 8.5pt;
      white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 8px 0; }
th, td { border: 1px solid #bbb; padding: 4px 6px; vertical-align: top; text-align: left; }
th { background: #eee; }
blockquote { border-left: 3px solid #999; margin-left: 0; padding-left: 10px; color: #333; }
"""


def main():
    src = pathlib.Path(sys.argv[1])
    body = markdown.markdown(src.read_text(), extensions=['extra', 'toc', 'sane_lists'])
    html = f'<html><head><meta charset="utf-8"><style>{CSS}</style></head><body>{body}</body></html>'
    out = src.with_suffix('.pdf')
    HTML(string=html, base_url=str(src.parent)).write_pdf(out)
    print(out)


if __name__ == '__main__':
    main()
