# -*- coding: utf-8 -*-
"""端到端流水线验证：求解 → 方案整合 → 规则检查（检查优先，违规即报错收集）。"""
import sys

import pandas as pd

sys.path.insert(0, r'd:\AAA_Jupyter\BBB_Competition\2025\C\src')

from de_solver_full import FullDE, YEARS, load_2023_baseline
from rule_checker import RuleChecker, integrate
from scheme import SchemeModel

sm = SchemeModel()
rc = RuleChecker(sm)
base = load_2023_baseline(sm)

for problem, mode in [(1, 1), (1, 2)]:
    de = FullDE(seed=0)
    plan, fit = de.solve(baseline=base, problem=problem, mode=mode,
                         npop=12, ngen=10)                 # 快速验证（非最终参数）
    # 检查优先：raise_if_invalid 在 solve 内已调用；此处独立复核一遍
    rep = rc.raise_if_invalid(integrate({y: plan[y] for y in YEARS}, base=base), base=base)
    print(f'问题{problem} mode{mode}: 收益 {fit/1e4:.2f} 万 → 规则检查 '
          f'{"全部通过" if rep.ok else f"违规 {len(rep.violations)}"}: {rep.summary()}')
