# -*- coding: utf-8 -*-
"""生成经济参数明细表：按 类型-季节编号 × 作物 逐行列出 亩产量/种植成本/售价区间/售价中值。
粒度 = 地块类型·季次（125 行 = 附件2统计表 107 行 + 补充智慧大棚第一季 18 行）。
类型-季节编号（字母=地块类型，数字=季节，单季无后缀）：
  A 平旱地·单季   B 梯田·单季    C 山坡地·单季
  D 水浇地·单季(水稻)  D-1 水浇地·第一季  D-2 水浇地·第二季
  E-1 普通大棚·第一季  E-2 普通大棚·第二季
  F-1 智慧大棚·第一季  F-2 智慧大棚·第二季
智慧大棚·第一季：附件2 说明"其亩产量、种植成本和销售价格均与普通大棚相同，表中省略"，
故 F-1 参数复制 E-1（作物 17~34，18 行）。
"""
import pandas as pd
import unicodedata

DOCS = r'd:\AAA_Jupyter\BBB_Competition\2025\C\docs'
OUT = r'd:\AAA_Jupyter\BBB_Competition\2025\C\data'

# 档位顺序与编号（单季无后缀）
BANDS = [('平旱地', '单季', 'A'), ('梯田', '单季', 'B'), ('山坡地', '单季', 'C'),
         ('水浇地', '单季', 'D'), ('水浇地', '第一季', 'D-1'), ('水浇地', '第二季', 'D-2'),
         ('普通大棚', '第一季', 'E-1'), ('普通大棚', '第二季', 'E-2'),
         ('智慧大棚', '第一季', 'F-1'), ('智慧大棚', '第二季', 'F-2')]
band_id = {(t, s): e for t, s, e in BANDS}


def price_mid(s):
    lo, hi = str(s).strip().split('-')
    return round((float(lo) + float(hi)) / 2, 2)


d = pd.read_excel(f'{DOCS}/附件2.xlsx', sheet_name='2023年统计的相关数据', header=None)
stat = d.iloc[1:108].copy()                     # 跳过表头，107 数据行
stat.columns = ['序号', '作物编号', '作物名称', '地块类型', '种植季次',
                '亩产量/斤', '种植成本/(元/亩)', '销售单价/(元/斤)']
stat = stat[pd.to_numeric(stat['序号'], errors='coerce').notna()]
stat['地块类型'] = stat['地块类型'].astype(str).str.strip()
stat['种植季次'] = stat['种植季次'].astype(str).str.strip()
stat['作物编号'] = pd.to_numeric(stat['作物编号']).astype(int)
stat['售价中值/(元/斤)'] = stat['销售单价/(元/斤)'].apply(price_mid)

df = stat[['地块类型', '种植季次', '作物编号', '作物名称',
           '亩产量/斤', '种植成本/(元/亩)', '销售单价/(元/斤)', '售价中值/(元/斤)']].copy()
df.columns = ['地块类型', '种植季次', '作物编号', '作物名称',
              '亩产量/斤', '种植成本/(元/亩)', '售价区间/(元/斤)', '售价中值/(元/斤)']
df['类型-季节编号'] = df.apply(lambda r: band_id[(r['地块类型'], r['种植季次'])], axis=1)

# 补充智慧大棚·第一季（F-1）：附件2 说明与普通大棚·第一季相同（表中省略），复制 E-1 参数
f1 = df[df['类型-季节编号'] == 'E-1'].copy()
f1['类型-季节编号'] = 'F-1'
f1['地块类型'] = '智慧大棚'
f1['种植季次'] = '第一季'
df = pd.concat([df, f1], ignore_index=True)

order = {e: i for i, (_, _, e) in enumerate(BANDS)}
df = df.sort_values(by=['类型-季节编号', '作物编号'], key=lambda s: s.map(order) if s.name == '类型-季节编号' else s)
df = df[['类型-季节编号', '地块类型', '种植季次', '作物编号', '作物名称',
         '亩产量/斤', '种植成本/(元/亩)', '售价区间/(元/斤)', '售价中值/(元/斤)']]


def disp_w(s):
    return sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in str(s))


def align_cells(header, rows_):
    data = [[str(v) for v in r] for r in rows_]
    w = [max(disp_w(h), *(disp_w(r[i]) for r in data)) for i, h in enumerate(header)]
    num = [all(r[i].replace('-', '', 1).replace('.', '', 1).isdigit() or r[i].endswith('%')
               for r in data) for i in range(len(header))]
    out = []
    for r in [list(map(str, header))] + data:
        cells = []
        for i, s in enumerate(r):
            if num[i]:
                cells.append(' ' * (w[i] - disp_w(s)) + s)
            else:
                cells.append(s + ' ' * (w[i] - disp_w(s)))
        out.append(', '.join(cells).rstrip())
    return out


with open(f'{OUT}/经济参数明细.csv', 'w', encoding='utf-8-sig', newline='') as f:
    f.write('\n'.join(align_cells(df.columns, df.values.tolist())) + '\n')
print(f'行数={len(df)}')
