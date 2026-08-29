# -*- coding: utf-8 -*-
"""把附件1、附件2中的"说明/注"原文逐字收集到一个 markdown 文件（不改变原文）。"""
import pandas as pd

DOCS = r'd:\AAA_Jupyter\BBB_Competition\2025\C\docs'
OUT = r'd:\AAA_Jupyter\BBB_Competition\2025\C\data'


def clean_lines(*cells):
    """合并若干单元格文本，按行拆分、剔除空行，其余逐字保留。"""
    lines = []
    for c in cells:
        if pd.isna(c):
            continue
        for ln in str(c).replace('\r', '').split('\n'):
            if ln.strip():
                lines.append(ln.rstrip())
    return lines


d1 = pd.read_excel(f'{DOCS}/附件1.xlsx', sheet_name='乡村的现有耕地', header=None)
d2 = pd.read_excel(f'{DOCS}/附件1.xlsx', sheet_name='乡村种植的农作物', header=None)
d3 = pd.read_excel(f'{DOCS}/附件2.xlsx', sheet_name='2023年统计的相关数据', header=None)

secs = ['# 附件注意事项与说明', '',
        '> 以下内容逐字摘录自 2024 年"高教社杯"全国大学生数学建模竞赛 C 题附件1、附件2 中的"说明"与"注"内容，未改动原文。', '']


def add_section(title, lines):
    secs.append(f'## {title}')
    secs.append('')
    for ln in lines:
        secs.append(f'> {ln}')
    secs.append('')


add_section('附件1 · 乡村的现有耕地（地块说明）', clean_lines(d1.iloc[1, 3]))
add_section('附件1 · 乡村种植的农作物（作物说明）', clean_lines(d2.iloc[1, 4]))
add_section('附件1 · 乡村种植的农作物（表尾注：各季节时间）',
            clean_lines(d2.iloc[43, 1], d2.iloc[44, 1], d2.iloc[45, 1]))
add_section('附件2 · 2023年统计的相关数据（表尾注）',
            clean_lines(d3.iloc[109, 1], d3.iloc[110, 1]))

path = f'{OUT}/附件注意事项与说明.md'
with open(path, 'w', encoding='utf-8-sig') as f:
    f.write('\n'.join(secs) + '\n')
print('[OK]', path)
