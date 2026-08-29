# -*- coding: utf-8 -*-
"""源数据加工：从 data/2023种植情况.csv × data/经济参数明细.csv 计算各作物 2023 总产量(=销售量)。

原始依据链：
  附件2《2023年的农作物种植情况》 → scripts/make_plant_2023.py → data/2023种植情况.csv（87 条真实耕种记录）
  档位级参数（亩产量/成本/售价）统一以 data/经济参数明细.csv 为准（已由 scripts/make_econ_detail.py 加工，
  含显式补充的智慧大棚·第一季 F-1 档，与普通大棚·第一季参数相同）。

口径：
- 2023 无销售量数据，按"产量=销量"近似（题目默认 2023 全部卖出）；
- 每种作物产量 = Σ_{2023 种植该作物的各地块} 种植面积 × 该地块档位亩产量；
- 档位 = (地块类型, 种植季次)。
"""
import pandas as pd

DATA = r'd:\AAA_Jupyter\BBB_Competition\2025\C\data'
DOCS = r'd:\AAA_Jupyter\BBB_Competition\2025\C\docs'


def crop_sales_2023():
    """返回 {作物编号(int): 2023总产量/斤(=销售量)}，作物级销量基准。"""
    # 档位 × 作物 → 亩产量（经济参数明细.csv，F-1 已显式存在，无需映射）
    det = pd.read_csv(f'{DATA}/经济参数明细.csv', skipinitialspace=True)
    det['地块类型'] = det['地块类型'].astype(str).str.strip()
    det['种植季次'] = det['种植季次'].astype(str).str.strip()
    det['作物编号'] = pd.to_numeric(det['作物编号']).astype(int)
    det['亩产量/斤'] = pd.to_numeric(det['亩产量/斤'])
    yield_lookup = {}
    for _, r in det.iterrows():
        yield_lookup[(r['地块类型'], r['种植季次'], r['作物编号'])] = r['亩产量/斤']

    # 附件1：地块名称 → 地块类型
    plots = pd.read_excel(f'{DOCS}/附件1.xlsx', sheet_name='乡村的现有耕地')
    plot_type = dict(zip(plots['地块名称'].astype(str).str.strip(),
                         plots['地块类型'].astype(str).str.strip()))

    # 2023 真实耕种情况（每块地·季次·作物·面积）
    plant = pd.read_csv(f'{DATA}/2023种植情况.csv', skipinitialspace=True)
    plant['种植地块'] = plant['种植地块'].astype(str).str.strip()
    plant['种植季次'] = plant['种植季次'].astype(str).str.strip()
    plant['种植面积/亩'] = pd.to_numeric(plant['种植面积/亩'])

    # 作物级聚合：Σ 面积 × 档位亩产量
    sales = {}
    for _, r in plant.iterrows():
        j = int(r['作物编号'])
        k = (plot_type[r['种植地块']], r['种植季次'], j)
        sales[j] = sales.get(j, 0.0) + float(r['种植面积/亩']) * yield_lookup[k]
    return {j: round(v, 1) for j, v in sales.items()}


if __name__ == '__main__':
    s = crop_sales_2023()
    print(f'作物数={len(s)}, 总产量={sum(s.values()):.1f} 斤')
    for j in range(1, 42):
        print(f'  作物{j}: {s[j]:.0f} 斤')
