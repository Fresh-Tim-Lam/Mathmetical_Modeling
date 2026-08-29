# -*- coding: utf-8 -*-
"""把附件2 的 2023 年真实耕种情况导出为 CSV，作为数据加工的唯一原始依据（"有根据"）：
  data/2023种植情况.csv   —— 2023年的农作物种植情况（87 条真实耕种记录，地块编号合并单元格已填满）
下游：scripts/make_sales_table.py（经济数据）、src/data_prep.py（销量基准）从此 CSV 读取种植情况；
      档位级参数（亩产量/成本/售价）统一以 data/经济参数明细.csv 为准（已由 make_econ_detail.py 加工）。
"""
import pandas as pd

DOCS = r'd:\AAA_Jupyter\BBB_Competition\2025\C\docs'
OUT = r'd:\AAA_Jupyter\BBB_Competition\2025\C\data'


def export_plant():
    d = pd.read_excel(f'{DOCS}/附件2.xlsx', sheet_name='2023年的农作物种植情况')
    d['种植地块'] = d['种植地块'].ffill()          # 合并单元格 → 填满
    d['种植地块'] = d['种植地块'].astype(str).str.strip()
    d['种植面积/亩'] = pd.to_numeric(d['种植面积/亩'])
    d.to_csv(f'{OUT}/2023种植情况.csv', index=False, encoding='utf-8-sig')
    print(f'2023种植情况.csv: {len(d)} 行')


if __name__ == '__main__':
    export_plant()
