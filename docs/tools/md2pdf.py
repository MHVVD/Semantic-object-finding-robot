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
import re
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
pre code { font-size: 7.4pt; }
pre code { background: none; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 8px 0; }
th, td { border: 1px solid #bbb; padding: 4px 6px; vertical-align: top; text-align: left; }
th { background: #eee; }
img { max-width: 100%; height: auto; display: block; margin: 6px auto; page-break-inside: avoid; }
blockquote { border-left: 3px solid #999; margin-left: 0; padding-left: 10px; color: #333; }
"""


LIST_ITEM = re.compile(r'^\s*(?:[-*+]|\d+\.)\s')


def indent(line):
    return len(line) - len(line.lstrip(' '))


def blank_line_before_lists(text):
    """
    Insert a blank line before a list that directly follows a paragraph line.

    Also before any list item that returns to an outer list (less indented than
    the line before it), which Python-Markdown would otherwise absorb into the
    nested item or paragraph above.

    GitHub renders such lists, Python-Markdown folds them into the paragraph.
    Fenced code blocks are left untouched.
    """
    out, in_code, prev = [], False, ''
    for line in text.splitlines():
        if line.lstrip().startswith('```'):
            in_code = not in_code
        elif (not in_code and LIST_ITEM.match(line) and prev.strip()
              and (indent(line) < indent(prev)
                   or (not LIST_ITEM.match(prev) and not prev.startswith(' ')))):
            out.append('')
        out.append(line)
        prev = line
    return '\n'.join(out) + '\n'


def widen_indents(text):
    """
    Double leading spaces outside fenced code blocks.

    GitHub nests a list under a 2-space indent; Python-Markdown needs 4, and
    otherwise flattens nested items into the parent list.
    """
    out, in_code, extra = [], False, 0
    for line in text.splitlines():
        if line.lstrip().startswith('```'):
            if not in_code:
                extra = indent(line)       # a fence inside a list item moves with it
            line = ' ' * extra + line
            in_code = not in_code
        elif in_code:
            line = ' ' * extra + line if line.strip() else line
        elif line.startswith(' '):
            line = ' ' * indent(line) + line
        out.append(line)
    return '\n'.join(out) + '\n'


def main():
    src = pathlib.Path(sys.argv[1])
    text = blank_line_before_lists(widen_indents(src.read_text()))
    body = markdown.markdown(text, extensions=['extra', 'toc', 'sane_lists'])
    html = f'<html><head><meta charset="utf-8"><style>{CSS}</style></head><body>{body}</body></html>'
    out = src.with_suffix('.pdf')
    HTML(string=html, base_url=str(src.parent)).write_pdf(out)
    print(out)


if __name__ == '__main__':
    main()
