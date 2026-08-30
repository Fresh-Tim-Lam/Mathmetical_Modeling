# -*- coding: utf-8 -*-
"""规则合规集成验证：当前求解器（全量权重 DE）产出的方案必须全部通过规则检查器。"""
import sys

sys.path.insert(0, r'd:\AAA_Jupyter\BBB_Competition\2025\C\src')

from de_solver_full import FullDE, YEARS, load_2023_baseline
from rule_checker import RuleChecker, integrate
from scheme import SchemeModel

BASE = r'd:\AAA_Jupyter\BBB_Competition\2025\C'
sm = SchemeModel()
rc = RuleChecker(sm)
base = load_2023_baseline(sm)

# 先验：2023 基线本身必须通过
rep23 = rc.check_year(base)
print(f'2023 基线 check: {"通过" if not rep23 else rep23}')

scenarios = [(1, 1, 'normal'), (1, 2, 'normal'), (2, 1, 'normal')]
for problem, mode, dist in scenarios:
    de = FullDE(seed=0)
    plan, fit = de.solve(baseline=base, problem=problem, mode=mode,
                         dist=dist, n_quad=16, npop=12, ngen=10)   # 快速验证
    # 检查优先：solve 内部已 raise_if_invalid；此处独立复核
    rep = rc.raise_if_invalid(integrate({y: plan[y] for y in YEARS}, base=base), base=base)
    status = 'OK' if rep.ok else 'FAIL'
    print(f'问题{problem} mode={mode}: {status}  违规={len(rep.violations)} '
          f'（{rep.summary()}）适应度={fit/1e4:.2f}万')
