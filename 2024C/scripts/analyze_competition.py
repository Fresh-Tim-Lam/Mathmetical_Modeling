# -*- coding: utf-8 -*-
"""问题 1/2 针对题目的分析：图片 + 数据 → outputs/ 文件夹。

数据来源（K=2 热启动 + 全量编码 DE 方案，见 src/de_solver_full.py __main__）：
  data/de_plan_full_p1_mode1.pkl   问题1 情形1（确定性，超量滞销 κ=0）
  data/de_plan_full_p1_mode2.pkl   问题1 情形2（确定性，超量半价 κ=0.5）
  data/de_plan_full_p2_mode1.pkl   问题2（期望收益 normal，κ=0 滞销口径）

输出（outputs/）：
  图片：p1_annual_revenue.png / p1_p2_utilization.png / p1_crop_composition.png
        p2_expected_revenue.png / p1_vs_p2.png
  数据：p1_p2_annual_revenue.csv / p1_p2_utilization.csv / crop_category_area.csv / summary.csv

口径：
  - 收益：问题1 确定性（profit_det）；问题2 期望（profit_stoch, normal, n_quad=16）。
  - 复种指数 = 年度种植面积（含两季）/ 耕地总面积（两季地块只计一季面积）。
  - 作物类别：豆类{1..5,17..19}、粮食(其他){6..15}、水稻{16}、蔬菜{20..37}、食用菌{38..41}。
"""
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(r'd:\AAA_Jupyter\BBB_Competition\2025\C')
sys.path.insert(0, str(BASE / 'src'))

from de_solver_full import FullDE, YEARS  # noqa: E402

OUT = BASE / 'outputs'
DATA = BASE / 'data'

# 中文字体（Windows）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

YEARS_L = list(YEARS)

CATS = [
    ('豆类', {1, 2, 3, 4, 5, 17, 18, 19}),
    ('粮食(其他)', set(range(6, 16))),
    ('水稻', {16}),
    ('蔬菜', set(range(20, 38))),
    ('食用菌', {38, 39, 40, 41}),
]
CAT_COLORS = ['#8da0cb', '#e8c463', '#66c2a5', '#a6d854', '#fc8d62']


def load_plan(name):
    with open(DATA / name, 'rb') as f:
        return pickle.load(f)


def crop_area(de, plan, y):
    """年度各作物种植面积（亩）。"""
    areas = defaultdict(float)
    for p, seasons in plan[y].items():
        for s, crops in seasons.items():
            au = de.sm.unit_area[de.sm.unit_of(p, s)]
            for j, a in crops:
                if a > 0:
                    areas[j] += a * au
    return dict(areas)


def cat_area(ca):
    """作物面积 dict → 类别面积 dict。"""
    out = {}
    for name, js in CATS:
        out[name] = sum(v for j, v in ca.items() if j in js)
    out['其他/未分类'] = sum(v for j, v in ca.items()
                          if not any(j in js for _, js in CATS))
    return out


def planted_area(de, plan, y):
    return sum(crop_area(de, plan, y).values())


def total_land(de):
    return sum(de.sm.unit_area[de.sm.unit_of(p, 1)] for p in de.sm.plots)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    de = FullDE(seed=0)
    land = total_land(de)

    p1a = load_plan('de_plan_full_p1_mode1.pkl')
    p1b = load_plan('de_plan_full_p1_mode2.pkl')
    p2 = load_plan('de_plan_full_p2_mode1.pkl')

    # 检查优先：三个方案先过规则检查硬门禁（违规即 raise）
    from scheme import SchemeModel
    sm = SchemeModel()
    plant = pd.read_csv(DATA / '2023种植情况.csv', skipinitialspace=True)
    bbase = defaultdict(lambda: defaultdict(list))
    for _, r in plant.iterrows():
        plot = str(r['种植地块']).strip()
        s = 1 if str(r['种植季次']).strip() in ('单季', '第一季') else 2
        u = sm.unit_of(plot, s)
        bbase[plot][s].append((int(r['作物编号']),
                               float(r['种植面积/亩']) / sm.unit_area[u]))
    bbase = dict(bbase)
    for tag, plan in (('p1_mode1', p1a), ('p1_mode2', p1b), ('p2', p2)):
        de.rc.raise_if_invalid({y: plan[y] for y in YEARS}, base=bbase)
        print(f'规则检查 {tag}: 通过 ✓')

    # ---------- 数据表 ----------
    rows = []
    for y in YEARS_L:
        p1a_y = de._profit(p1a, y, 1, 1, 'normal', 16) / 1e4
        p1b_y = de._profit(p1b, y, 1, 2, 'normal', 16) / 1e4
        p2_y = de._profit(p2, y, 2, 1, 'normal', 16) / 1e4
        rows.append({'年份': y, '问题1情形1/万元': p1a_y,
                     '问题1情形2/万元': p1b_y, '问题2期望/万元': p2_y})
    df_rev = pd.DataFrame(rows)
    df_rev.to_csv(OUT / 'p1_p2_annual_revenue.csv', index=False, encoding='utf-8-sig')

    # 利用情况
    util = []
    for y in YEARS_L:
        util.append({'年份': y,
                     '情形1种植/亩': planted_area(de, p1a, y),
                     '情形1复种指数': planted_area(de, p1a, y) / land,
                     '情形2种植/亩': planted_area(de, p1b, y),
                     '情形2复种指数': planted_area(de, p1b, y) / land,
                     '问题2种植/亩': planted_area(de, p2, y),
                     '问题2复种指数': planted_area(de, p2, y) / land})
    df_util = pd.DataFrame(util)
    df_util.to_csv(OUT / 'p1_p2_utilization.csv', index=False, encoding='utf-8-sig')

    # 作物面积（长表）
    cat_rows = []
    for y in YEARS_L:
        for tag, plan in (('情形1', p1a), ('情形2', p1b), ('问题2', p2)):
            ca = cat_area(crop_area(de, plan, y))
            for name, a in ca.items():
                cat_rows.append({'年份': y, '方案': tag, '作物类别': name, '面积/亩': round(a, 2)})
    pd.DataFrame(cat_rows).to_csv(OUT / 'crop_category_area.csv', index=False,
                                  encoding='utf-8-sig')

    # 汇总
    tot = {
        '问题1情形1总收益/万元': de.fitness(p1a, problem=1, mode=1, base=bbase) / 1e4,
        '问题1情形2总收益/万元': de.fitness(p1b, problem=1, mode=2, base=bbase) / 1e4,
        '问题2期望总收益/万元': de.fitness(p2, problem=2, mode=1, dist='normal',
                                       n_quad=16, base=bbase) / 1e4,
        '耕地总面积/亩': land,
        '情形1平均复种指数': df_util['情形1复种指数'].mean(),
        '情形2平均复种指数': df_util['情形2复种指数'].mean(),
    }
    pd.DataFrame([tot]).T.to_csv(OUT / 'summary.csv', encoding='utf-8-sig',
                                 header=['数值'])

    # ---------- 图片 ----------
    x = np.arange(len(YEARS_L))
    w = 0.35

    # 1) 问题1 年度收益（两情形）
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, df_rev['问题1情形1/万元'], w, label='情形1（超量滞销 κ=0）', color='#4c72b0')
    ax.bar(x + w / 2, df_rev['问题1情形2/万元'], w, label='情形2（超量半价 κ=0.5）', color='#dd8452')
    for i, (v1, v2) in enumerate(zip(df_rev['问题1情形1/万元'], df_rev['问题1情形2/万元'])):
        ax.text(i - w / 2, v1 + 1, f'{v1:.0f}', ha='center', fontsize=8)
        ax.text(i + w / 2, v2 + 1, f'{v2:.0f}', ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(YEARS_L)
    ax.set_xlabel('年份')
    ax.set_ylabel('确定性收益（万元）')
    ax.set_title('问题1：2024-2030 年度收益（两种情形对比）')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / 'p1_annual_revenue.png', dpi=150)
    plt.close(fig)

    # 2) 复种指数
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(YEARS_L, df_util['情形1复种指数'], 'o-', label='情形1（超量滞销 κ=0）', color='#4c72b0')
    ax.plot(YEARS_L, df_util['情形2复种指数'], 's-', label='情形2（超量半价 κ=0.5）', color='#dd8452')
    ax.plot(YEARS_L, df_util['问题2复种指数'], '^-', label='问题2', color='#55a868')
    ax.set_xlabel('年份')
    ax.set_ylabel('复种指数（种植面积/耕地面积）')
    ax.set_title('问题1/2：土地利用强度（复种指数）')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / 'p1_p2_utilization.png', dpi=150)
    plt.close(fig)

    # 3) 作物构成（情形1/情形2 堆叠面积）
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (tag, plan) in zip(axes, (('情形1', p1a), ('情形2', p1b))):
        mat = np.zeros((len(YEARS_L), len(CATS)))
        for i, y in enumerate(YEARS_L):
            ca = cat_area(crop_area(de, plan, y))
            for k, (name, _) in enumerate(CATS):
                mat[i, k] = ca[name]
        bottom = np.zeros(len(YEARS_L))
        for k, (name, _) in enumerate(CATS):
            ax.bar(x, mat[:, k], bottom=bottom, label=name, color=CAT_COLORS[k])
            bottom += mat[:, k]
        ax.set_xticks(x)
        ax.set_xticklabels(YEARS_L)
        ax.set_xlabel('年份')
        ax.set_ylabel('面积（亩）')
        ax.set_title(f'问题1 {tag}：作物构成')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'p1_crop_composition.png', dpi=150)
    plt.close(fig)

    # 4) 问题2 期望收益
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, df_rev['问题2期望/万元'], color='#55a868')
    for i, v in enumerate(df_rev['问题2期望/万元']):
        ax.text(i, v + 1, f'{v:.0f}', ha='center', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(YEARS_L)
    ax.set_xlabel('年份')
    ax.set_ylabel('期望收益（万元）')
    ax.set_title('问题2：2024-2030 年度期望收益（normal 分布）')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / 'p2_expected_revenue.png', dpi=150)
    plt.close(fig)

    # 5) 问题1情形1 确定性 vs 问题2 期望
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(YEARS_L, df_rev['问题1情形1/万元'], 'o-', label='问题1 情形1（确定性）', color='#4c72b0')
    ax.plot(YEARS_L, df_rev['问题2期望/万元'], 's--', label='问题2（期望收益）', color='#55a868')
    ax.set_xlabel('年份')
    ax.set_ylabel('收益（万元）')
    ax.set_title('问题1情形1 与 问题2 对比：确定性 vs 期望收益')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / 'p1_vs_p2.png', dpi=150)
    plt.close(fig)

    print(f'\n输出目录: {OUT}')
    for f in sorted(OUT.iterdir()):
        print(f'  {f.name}')


if __name__ == '__main__':
    main()
