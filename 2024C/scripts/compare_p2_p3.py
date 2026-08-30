# -*- coding: utf-8 -*-
"""问题2 vs 问题3 比较分析（"通过模拟数据进行求解，并与问题2的结果作比较分析"）。

同一套 CRN 相关样本（N=20000，R=corr_matrix 相关结构）下，对 p2/p3 两方案评估：
- 问题3 框架（相关 MC）：期望收益、风险（年/合计 std、CVaR5%）；
- 问题2 框架（独立分布求积 profit_stoch，normal，n_quad=16）：期望收益；
- 方案差异：逐年作物级种植面积差。

输出：outputs/p3_compare.csv（收益/风险逐年+合计比较）、outputs/p3_area_diff.csv（面积差异）。
用法：python compare_p2_p3.py（在 scripts/ 下运行，需先完成 p2/p3 求解并保存 pkl）。
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r'd:\AAA_Jupyter\BBB_Competition\2025\C')
sys.path.insert(0, str(BASE / 'src'))

from de_solver_full import YEARS, load_2023_baseline
from revenue import RevenueModel
from scheme import SchemeModel

N_FINAL = 20000      # 最终评估情景数（低 MC 噪声）
N_QUAD = 16          # 问题2 求积阶数（与 run_all.py 口径一致）
SEED = 0


def area_by_crop(plan, y, sm):
    """逐年作物级种植面积（亩）：{作物编号: 面积}。"""
    area = {}
    for p, seasons in plan[y].items():
        for s, cs_ in seasons.items():
            u = sm.unit_of(p, s)
            a_u = sm.unit_area[u]
            for j, alpha in cs_:
                area[j] = area.get(j, 0.0) + alpha * a_u
    return area


def cvar5(pi):
    """5% 尾部风险：收益向量 π 的 CVaR = mean(最小的 5%)。"""
    k = max(1, int(round(len(pi) * 0.05)))
    return float(np.sort(pi)[:k].mean())


def main():
    rev = RevenueModel()
    sm = SchemeModel()
    R = rev.corr_matrix()
    base = load_2023_baseline(sm)
    with open(BASE / 'data' / 'de_plan_full_p2_mode1.pkl', 'rb') as f:
        p2 = pickle.load(f)
    with open(BASE / 'data' / 'de_plan_full_p3_mode1.pkl', 'rb') as f:
        p3 = pickle.load(f)

    # CRN 大样本（逐年缓存；mc_samples 不依赖年份 → 各年同一组样本）
    samples = {y: rev.mc_samples(y, N=N_FINAL, R=R, seed=SEED) for y in YEARS}

    rows, diff_rows = [], []
    tot = {'E2_2': 0.0, 'E2_3': 0.0}
    for y in YEARS:
        x2 = sm.derive(p2[y]); x3 = sm.derive(p3[y])
        # 问题2 框架（独立求积）
        e2_2 = rev.profit_stoch(x2, y, dist='normal', mode=1, n_quad=N_QUAD)
        e2_3 = rev.profit_stoch(x3, y, dist='normal', mode=1, n_quad=N_QUAD)
        tot['E2_2'] += e2_2; tot['E2_3'] += e2_3
        # 问题3 框架（相关 MC，同一组 CRN 样本）
        pi2 = rev.profit_mc(x2, y, mode=1, N=N_FINAL, R=R, seed=SEED, samples=samples[y])
        pi3 = rev.profit_mc(x3, y, mode=1, N=N_FINAL, R=R, seed=SEED, samples=samples[y])
        a2, a3 = area_by_crop(p2, y, sm), area_by_crop(p3, y, sm)
        for j in sorted(set(a2) | set(a3)):
            diff_rows.append({'年份': y, '作物编号': j,
                              'p2面积/亩': a2.get(j, 0.0), 'p3面积/亩': a3.get(j, 0.0),
                              '差异(p3-p2)/亩': a3.get(j, 0.0) - a2.get(j, 0.0)})
        rows.append({'年': y,
                     'Q2框架期望(p2)/万元': e2_2 / 1e4, 'Q2框架期望(p3)/万元': e2_3 / 1e4,
                     'Q3-MC均值(p2)/万元': pi2.mean() / 1e4, 'Q3-MC均值(p3)/万元': pi3.mean() / 1e4,
                     'Q3-std(p2)/万元': pi2.std() / 1e4, 'Q3-std(p3)/万元': pi3.std() / 1e4,
                     'Q3-CVaR5(p2)/万元': cvar5(pi2) / 1e4, 'Q3-CVaR5(p3)/万元': cvar5(pi3) / 1e4,
                     'p2面积/亩': sum(a2.values()), 'p3面积/亩': sum(a3.values())})
    # 合计：7 年 MC 情景向量求和 → 总收益分布
    pi2t = sum(rev.profit_mc(sm.derive(p2[y]), y, mode=1, N=N_FINAL, R=R, seed=SEED,
                             samples=samples[y]) for y in YEARS)
    pi3t = sum(rev.profit_mc(sm.derive(p3[y]), y, mode=1, N=N_FINAL, R=R, seed=SEED,
                             samples=samples[y]) for y in YEARS)
    rows.append({'年': '合计',
                 'Q2框架期望(p2)/万元': tot['E2_2'] / 1e4, 'Q2框架期望(p3)/万元': tot['E2_3'] / 1e4,
                 'Q3-MC均值(p2)/万元': pi2t.mean() / 1e4, 'Q3-MC均值(p3)/万元': pi3t.mean() / 1e4,
                 'Q3-std(p2)/万元': pi2t.std() / 1e4, 'Q3-std(p3)/万元': pi3t.std() / 1e4,
                 'Q3-CVaR5(p2)/万元': cvar5(pi2t) / 1e4, 'Q3-CVaR5(p3)/万元': cvar5(pi3t) / 1e4,
                 'p2面积/亩': np.nan, 'p3面积/亩': np.nan})

    out = BASE / 'outputs'
    out.mkdir(exist_ok=True)
    cmp = pd.DataFrame(rows)
    cmp.to_csv(out / 'p3_compare.csv', index=False, encoding='utf-8-sig', float_format='%.2f')
    diff = pd.DataFrame(diff_rows)
    diff.to_csv(out / 'p3_area_diff.csv', index=False, encoding='utf-8-sig')
    print(cmp.to_string(index=False))
    print(f'\n面积差异（按年·作物，差异≠0 的记录数）：'
          f'{((diff["差异(p3-p2)/亩"]).abs() > 1e-6).sum()} / {len(diff)}')
    # 合计差异显著项（前 10 大变动）
    big = diff.assign(abs=diff['差异(p3-p2)/亩'].abs()).nlargest(10, 'abs')
    print('\n种植面积差异最大的 10 项（作物编号，p2→p3）：')
    for _, r in big.iterrows():
        print(f"  {int(r['年份'])} 作物{r['作物编号']:>2}: "
              f"{r['p2面积/亩']:.1f} → {r['p3面积/亩']:.1f} 亩（{r['差异(p3-p2)/亩']:+.1f}）")


if __name__ == '__main__':
    main()
