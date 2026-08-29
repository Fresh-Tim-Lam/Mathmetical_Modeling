# -*- coding: utf-8 -*-
"""生成 2023 作物产量(=销售量)汇总与波动规则表（经济数据.csv）。
销售量无原始数据，按题目说明直接用 2023 产量（种植面积 × 档位亩产量），
依据链：data/2023种植情况.csv（附件2 真实耕种记录）× data/经济参数明细.csv（档位参数）。
售价/成本因地块类型·季节而异，不做作物级汇总（统一以 data/经济参数明细.csv 为准），本表只留波动率。
"""
import pandas as pd

DOCS = r'd:\AAA_Jupyter\BBB_Competition\2025\C\docs'
OUT = r'd:\AAA_Jupyter\BBB_Competition\2025\C\data'

# ---------- 附件1：地块名称 → 地块类型 ----------
plots = pd.read_excel(f'{DOCS}/附件1.xlsx', sheet_name='乡村的现有耕地')
plot_type = dict(zip(plots['地块名称'].astype(str).str.strip(),
                     plots['地块类型'].astype(str).str.strip()))

# ---------- 附件1：作物编号 → 名称/类型 ----------
crops = pd.read_excel(f'{DOCS}/附件1.xlsx', sheet_name='乡村种植的农作物')
crops = crops[pd.to_numeric(crops['作物编号'], errors='coerce').notna()]
crop_name = dict(zip(pd.to_numeric(crops['作物编号']).astype(int),
                     crops['作物名称'].astype(str).str.strip()))
crop_cat = dict(zip(pd.to_numeric(crops['作物编号']).astype(int),
                    crops['作物类型'].astype(str).str.strip()))

# ---------- 档位亩产量（经济参数明细.csv，F-1 已显式存在，无需映射） ----------
det = pd.read_csv(f'{OUT}/经济参数明细.csv', skipinitialspace=True)
det['地块类型'] = det['地块类型'].astype(str).str.strip()
det['种植季次'] = det['种植季次'].astype(str).str.strip()
det['作物编号'] = pd.to_numeric(det['作物编号']).astype(int)
det['亩产量/斤'] = pd.to_numeric(det['亩产量/斤'])
yield_lookup = {}
for _, r in det.iterrows():
    yield_lookup[(r['地块类型'], r['种植季次'], r['作物编号'])] = r['亩产量/斤']

# ---------- 2023 种植情况（面积×亩产量 = 产量 = 销售量）→ CSV ----------
plant = pd.read_csv(f'{OUT}/2023种植情况.csv', skipinitialspace=True)
plant['种植地块'] = plant['种植地块'].astype(str).str.strip()
plant['种植季次'] = plant['种植季次'].astype(str).str.strip()
plant['种植面积/亩'] = pd.to_numeric(plant['种植面积/亩'])

sales, area_s = {}, {}
for _, r in plant.iterrows():
    j = int(r['作物编号'])
    k = (plot_type[r['种植地块']], r['种植季次'], j)
    area = r['种植面积/亩']
    sales[j] = sales.get(j, 0.0) + area * yield_lookup[k]
    area_s[j] = area_s.get(j, 0.0) + area

# ---------- 问题2 波动规则（docs/C题.md） ----------
YIELD_CHG = '±10%'      # 亩产量每年 ±10%
COST_CHG = '+5%'        # 种植成本平均每年增长 5%


def sales_chg(name):
    """预期销售量：小麦/玉米 增长 5%~10%，其他 ±5%。"""
    return '+5%~+10%' if name in ('小麦', '玉米') else '±5%'


def price_chg(cat, name):
    """销售价格：粮食稳定；蔬菜 +5%；食用菌 -1%~-5%；羊肚菌 -5%。"""
    if name == '羊肚菌':
        return '-5%'
    if cat == '食用菌':
        return '-1%~-5%'
    if cat.startswith('蔬菜'):
        return '+5%'
    return '0%'          # 粮食类基本稳定

# ---------- 汇总输出 ----------
rows = []
for j in range(1, 42):
    name, cat = crop_name[j], crop_cat[j]
    area = area_s.get(j, 0.0)
    rows.append([j, name, cat,
                 round(area, 2), round(sales.get(j, 0.0), 1),
                 sales_chg(name), YIELD_CHG, COST_CHG, price_chg(cat, name)])
df = pd.DataFrame(rows, columns=['作物编号', '作物名称', '作物类型',
                                 '2023种植面积/亩', '2023产量/斤(=销售量)',
                                 '预期销售量年变化率', '亩产量年变化率',
                                 '种植成本年增长率', '销售价格年变化率'])
import unicodedata


def disp_w(s):
    """字符串显示宽度：中文/全角字符按 2 格、ASCII 按 1 格。"""
    return sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in str(s))


def align_cells(header, rows):
    """按显示宽度对齐：数值列右对齐（表头随列右对齐）、文本列左对齐；仍用 ', ' 分隔（可导入）。"""
    data = [[str(v) for v in r] for r in rows]
    w = [max(disp_w(h), *(disp_w(r[i]) for r in data)) for i, h in enumerate(header)]
    num = [all(r[i].replace('-', '', 1).replace('.', '', 1).isdigit() or r[i].endswith('%')
               for r in data) for i in range(len(header))]
    out = []
    for r in [list(map(str, header))] + data:
        cells = []
        for i, s in enumerate(r):
            if num[i]:
                cells.append(' ' * (w[i] - disp_w(s)) + s)   # 数值右对齐
            else:
                cells.append(s + ' ' * (w[i] - disp_w(s)))   # 文本左对齐
        out.append(', '.join(cells).rstrip())
    return out


with open(f'{OUT}/经济数据.csv', 'w', encoding='utf-8-sig', newline='') as f:
    f.write('\n'.join(align_cells(df.columns, df.values.tolist())) + '\n')
print(df.to_string())
