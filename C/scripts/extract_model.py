# -*- coding: utf-8 -*-
"""只生成两个码表：
  1) 作物表.csv —— 主码 = 作物编号（含名称列）
  2) 地块表.csv —— 主码 = 地块编号（含季节编号 -1/-2，含名称列）
作物表的"可种植地块-季节编号"列出该作物可种植的各地块编号（见 地块表）。
CSV 分隔符为 ', '（逗号后加空格），便于人工阅读，同时仍可直接导入。
"""
import pandas as pd

DOCS = r'd:\AAA_Jupyter\BBB_Competition\2025\C\docs'
OUT = r'd:\AAA_Jupyter\BBB_Competition\2025\C\data'


def write_csv(path, df):
    lines = [', '.join(map(str, df.columns))]
    for r in df.values:
        lines.append(', '.join('' if pd.isna(v) else str(v) for v in r))
    with open(path, 'w', encoding='utf-8-sig') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'[OK] {path}  ({len(df)} 行)')


# ============ 地块类型 × 季节 参数 ============
SEASON_TIME = {
    ('平旱地', '单季'): ('全年一季', '粮食类（含豆类）'),
    ('梯田', '单季'): ('全年一季', '粮食类（含豆类）'),
    ('山坡地', '单季'): ('全年一季', '粮食类（含豆类）'),
    ('水浇地', '第一季'): ('3-6月', '水稻 或 蔬菜（大白菜/白萝卜/红萝卜除外）；也可单季种水稻'),
    ('水浇地', '第二季'): ('7-10月', '仅大白菜/白萝卜/红萝卜（择一种，便于管理）'),
    ('普通大棚', '第一季'): ('5-9月', '蔬菜（大白菜/白萝卜/红萝卜除外）'),
    ('普通大棚', '第二季'): ('9月-次年4月', '仅食用菌'),
    ('智慧大棚', '第一季'): ('3-7月', '蔬菜（大白菜/白萝卜/红萝卜除外）'),
    ('智慧大棚', '第二季'): ('8月-次年2月', '蔬菜（大白菜/白萝卜/红萝卜除外）'),
}

# 每个 地块类型×季次 可种的作物编号（逐个列出，格内用 "; " 分隔）
PLOT_CROPS = {
    ('平旱地', '单季'): list(range(1, 16)),
    ('梯田', '单季'): list(range(1, 16)),
    ('山坡地', '单季'): list(range(1, 16)),
    ('水浇地', '第一季'): [16] + list(range(17, 35)),
    ('水浇地', '第二季'): list(range(35, 38)),
    ('普通大棚', '第一季'): list(range(17, 35)),
    ('普通大棚', '第二季'): list(range(38, 42)),
    ('智慧大棚', '第一季'): list(range(17, 35)),
    ('智慧大棚', '第二季'): list(range(17, 35)),
}


def type_season_name(land_type, season):
    time, _ = SEASON_TIME[(land_type, season)]
    return f'{land_type}·{season}（{time}）'


def plot_seasons(land_type):
    if land_type in ('平旱地', '梯田', '山坡地'):
        return ['单季']
    return ['第一季', '第二季']


def plot_unit_codes(plot_name, land_type):
    """物理地块 → 地块编号：单季不带后缀，两季用 -1/-2。"""
    seasons = plot_seasons(land_type)
    if len(seasons) == 1:
        return [plot_name]
    return [f'{plot_name}-{i+1}' for i in range(len(seasons))]


def crop_ts(cat, name):
    """作物 → 可种植的 地块类型×季次 列表。"""
    if name == '水稻':
        return [('水浇地', '第一季')]
    if cat in ('粮食（豆类）', '粮食'):
        return [('平旱地', '单季'), ('梯田', '单季'), ('山坡地', '单季')]
    if cat in ('蔬菜（豆类）', '蔬菜'):
        if name in ('大白菜', '白萝卜', '红萝卜'):
            return [('水浇地', '第二季')]
        return [('水浇地', '第一季'), ('普通大棚', '第一季'),
                ('智慧大棚', '第一季'), ('智慧大棚', '第二季')]
    if cat == '食用菌':
        return [('普通大棚', '第二季')]
    return []


# ============ 读原始数据 ============
plots = pd.read_excel(f'{DOCS}/附件1.xlsx', sheet_name='乡村的现有耕地', header=None).iloc[1:55, :3]
plots.columns = ['地块名称', '地块类型', '地块面积/亩']
plots['地块名称'] = plots['地块名称'].str.strip()
plots['地块类型'] = plots['地块类型'].str.strip()
plots['地块面积/亩'] = pd.to_numeric(plots['地块面积/亩'])

crops = pd.read_excel(f'{DOCS}/附件1.xlsx', sheet_name='乡村种植的农作物', header=None).iloc[1:42, :4].copy()
crops.columns = ['作物编号', '作物名称', '作物类型', '种植耕地']
crops['作物编号'] = pd.to_numeric(crops['作物编号'])
for c in ('作物名称', '作物类型'):
    crops[c] = crops[c].str.strip()


# 地块类型×季次 → 该场景下的全部地块编号（与地块表一致）
ts_plots = {}
for _, p in plots.iterrows():
    for s, code in zip(plot_seasons(p['地块类型']), plot_unit_codes(p['地块名称'], p['地块类型'])):
        ts_plots.setdefault((p['地块类型'], s), []).append(code)


# ============ 表1：作物表（主码 = 作物编号） ============
crop_rows = []
for _, r in crops.iterrows():
    plant_codes = '; '.join(c for ts in crop_ts(r['作物类型'], r['作物名称']) for c in ts_plots[ts])
    crop_rows.append([r['作物编号'], r['作物名称'], r['作物类型'], plant_codes])
crop_dict = pd.DataFrame(crop_rows, columns=['作物编号', '作物名称', '作物类型', '可种植地块-季节编号'])
write_csv(f'{OUT}/作物表.csv', crop_dict)

# ============ 表2：地块表（主码 = 地块编号，含季节编号） ============
plot_rows = []
for _, p in plots.iterrows():
    for s, code in zip(plot_seasons(p['地块类型']), plot_unit_codes(p['地块名称'], p['地块类型'])):
        plot_rows.append([code, type_season_name(p['地块类型'], s),
                          p['地块面积/亩'],
                          '; '.join(str(c) for c in PLOT_CROPS[(p['地块类型'], s)])])
plot_dict = pd.DataFrame(plot_rows, columns=['地块编号', '名称', '面积/亩', '可种植作物编号'])
write_csv(f'{OUT}/地块表.csv', plot_dict)

print('\n全部完成，输出目录:', OUT)
