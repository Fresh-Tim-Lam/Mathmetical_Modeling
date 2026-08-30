# -*- coding: utf-8 -*-
"""全情景运行：问题1（mode1/2）+ 问题2（四种分布），输出年度收益与违规。"""
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, r'd:\AAA_Jupyter\BBB_Competition\2025\C\src')

from de_solver_full import FullDE
from revenue import DISTS
from scheme import SchemeModel

sm = SchemeModel()
plant = pd.read_csv(r'd:\AAA_Jupyter\BBB_Competition\2025\C\data\2023种植情况.csv',
                    skipinitialspace=True)
base = defaultdict(lambda: defaultdict(list))
for _, r in plant.iterrows():
    plot = str(r['种植地块']).strip()
    s = 1 if str(r['种植季次']).strip() in ('单季', '第一季') else 2
    u = sm.unit_of(plot, s)
    alpha = float(r['种植面积/亩']) / sm.unit_area[u]
    base[plot][s].append((int(r['作物编号']), alpha))
base = dict(base)

for mode in (1, 2):
    de = FullDE(seed=0)
    print(f'=== 问题1 mode={mode}（{"滞销" if mode == 1 else "半价"}）===')
    plan, fit = de.solve(baseline=base, problem=1, mode=mode, seed=0)
    de.report(plan, mode=mode)
    print(f'最优适应度: {fit / 1e4:.2f} 万（7 年合计）\n')

de = FullDE(seed=0)
print('=== 问题2 mode=1（期望收益，四种分布对比，n_quad=16）===')
for dist in DISTS:
    plan, fit = de.solve(baseline=base, problem=2, mode=1, dist=dist, seed=0, n_quad=16)
    de.report(plan, mode=1)
    print(f'[{dist}] 最优适应度: {fit / 1e4:.2f} 万（7 年合计，n_quad=16）\n')
