# -*- coding: utf-8 -*-
"""验证 经济数据.csv 的 2023产量/斤(=销售量)（智慧大棚·第一季 F-1 已显式入表后）：
三种口径交叉对比：
  ① 经济数据.csv 生成列（make_sales_table.py）
  ② pandas 合并独立重算：2023种植情况.csv × 经济参数明细.csv（档位键 (地块类型,季次,作物)
     直接匹配，含显式补充的 F-1 行）
  ③ src/data_prep.crop_sales_2023()（收益模块实际使用的作物级销量基准）
"""
import sys
import pandas as pd

BASE = r'd:\AAA_Jupyter\BBB_Competition\2025\C'
sys.path.insert(0, f'{BASE}\\src')
DATA = f'{BASE}\\data'
DOCS = f'{BASE}\\docs'

from data_prep import crop_sales_2023

# ---------- ② pandas 合并独立重算 ----------
det = pd.read_csv(f'{DATA}/经济参数明细.csv', skipinitialspace=True)
det['地块类型'] = det['地块类型'].astype(str).str.strip()
det['种植季次'] = det['种植季次'].astype(str).str.strip()
det['作物编号'] = pd.to_numeric(det['作物编号']).astype(int)
det['亩产量/斤'] = pd.to_numeric(det['亩产量/斤'])

plots = pd.read_excel(f'{DOCS}/附件1.xlsx', sheet_name='乡村的现有耕地')
plot_type = dict(zip(plots['地块名称'].astype(str).str.strip(),
                     plots['地块类型'].astype(str).str.strip()))

plant = pd.read_csv(f'{DATA}/2023种植情况.csv', skipinitialspace=True)
plant['种植地块'] = plant['种植地块'].astype(str).str.strip()
plant['种植季次'] = plant['种植季次'].astype(str).str.strip()
plant['地块类型'] = plant['种植地块'].map(plot_type)
plant['种植面积/亩'] = pd.to_numeric(plant['种植面积/亩'])

m = plant.merge(det, on=['地块类型', '种植季次', '作物编号'], how='inner')
m['产量/斤'] = m['种植面积/亩'] * m['亩产量/斤']
recalc = m.groupby('作物编号')['产量/斤'].sum()

# ---------- ① 经济数据.csv ----------
ec = pd.read_csv(f'{DATA}/经济数据.csv', skipinitialspace=True)
ec['作物编号'] = pd.to_numeric(ec['作物编号'])

# ---------- ③ data_prep ----------
prep = crop_sales_2023()

print('作物  CSV生成列      合并重算       data_prep      max差异')
bad = 0
for j in range(1, 42):
    v1 = float(ec.loc[ec['作物编号'] == j, '2023产量/斤(=销售量)'].iloc[0])
    v2 = float(recalc[j])
    v3 = float(prep[j])
    d = max(abs(v1 - v2), abs(v1 - v3), abs(v2 - v3))
    flag = '  <-- 不一致!' if d > 1e-6 else ''
    if d > 1e-6:
        bad += 1
    print(f'{j:>3}  {v1:>12.1f}  {v2:>12.1f}  {v3:>12.1f}  {d:>9.2e}{flag}')
print(f'\n总计: 总产量={recalc.sum():.0f} 斤, 不一致作物数={bad}')
