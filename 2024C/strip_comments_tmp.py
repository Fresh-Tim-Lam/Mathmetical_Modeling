# -*- coding: utf-8 -*-
"""临时脚本：剥离 src/ 下 Python 源码的注释与 docstring，输出到 code_submit/。"""
import ast
import io
import re
import tokenize
from pathlib import Path

SRC = Path(r'd:\AAA_Jupyter\BBB_Competition\Mathmetical_Modeling\2024C\src')
DST = Path(r'd:\AAA_Jupyter\BBB_Competition\Mathmetical_Modeling\2024C\code_submit')


def analyze(src_text):
    """返回 (docstring 行号集合, 需补 pass 的位置列表 [(行号, 缩进)])。"""
    doc = set()
    need_pass = []
    try:
        tree = ast.parse(src_text)
    except SyntaxError as e:
        print('  [警告] ast.parse 失败:', e)
        return doc, need_pass
    for node in ast.walk(tree):
        if isinstance(node, ast.Module):
            body = node.body
            if body and isinstance(body[0], ast.Expr) \
                    and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                v = body[0].value
                for ln in range(v.lineno, (v.end_lineno or v.lineno) + 1):
                    doc.add(ln)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) \
                    and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                v = body[0].value
                end = v.end_lineno or v.lineno
                for ln in range(v.lineno, end + 1):
                    doc.add(ln)
                if not body[1:]:
                    line = src_text.splitlines()[v.lineno - 1]
                    indent = line[:len(line) - len(line.lstrip())]
                    need_pass.append((end, indent))
    return doc, need_pass


def strip(src_text):
    doc, need_pass = analyze(src_text)
    comment_col = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src_text).readline):
            if tok.type == tokenize.COMMENT:
                comment_col.setdefault(tok.start[0], tok.start[1])
    except tokenize.TokenError as e:
        print('  [警告] tokenize 失败:', e)
    lines = src_text.splitlines(keepends=True)
    out = []
    for i, line in enumerate(lines, start=1):
        if i in doc:
            for end_line, indent in need_pass:
                if i == end_line:
                    out.append(indent + 'pass\n')
            continue
        col = comment_col.get(i)
        if col is not None:
            line = line[:col].rstrip() + '\n'
        if line.strip() == '':
            if out and out[-1].strip() == '':
                continue
        out.append(line)
    text = ''.join(out)
    text = re.sub(r'[ \t]+$', '', text, flags=re.M)
    if not text.endswith('\n'):
        text += '\n'
    return text


def main():
    DST.mkdir(parents=True, exist_ok=True)
    for f in sorted(SRC.glob('*.py')):
        src = f.read_text(encoding='utf-8')
        out = strip(src)
        (DST / f.name).write_text(out, encoding='utf-8')
        print(f'{f.name}: {len(src.splitlines())} -> {len(out.splitlines())} 行')


if __name__ == '__main__':
    main()
