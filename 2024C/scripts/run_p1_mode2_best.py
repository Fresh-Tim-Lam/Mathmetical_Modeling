# -*- coding: utf-8 -*-
"""问题1 情形2（半价）多种子 32×80 求解，取最优覆盖基准解。

背景：p1_seed_stability.csv 显示情形2 半预算(32×40)解全部高于 32×80 基准 6198.30，
说明基准非收敛极限。本脚本以 32×80 × seeds 1..4 重解，保留最优，备份旧基准。
输出 outputs/p1_mode2_best.csv；覆盖 data/de_plan_full_p1_mode2.pkl。
用法：python run_p1_mode2_best.py（在 scripts/ 下）。
"""
import pickle
import shutil
import sys
from pathlib import Path

import pandas as pd

BASE = Path(r'd:\AAA_Jupyter\BBB_Competition\2025\C')
sys.path.insert(0, str(BASE / 'src'))

from de_solver_full import FullDE, YEARS, integrate, load_2023_baseline

NPOP, NGEN, SEEDS = 32, 80, (1, 2, 3, 4)
PKL = BASE / 'data' / 'de_plan_full_p1_mode2.pkl'


def solve_once(seed):
    base = load_2023_baseline()
    de = FullDE(seed=seed)
    plan, fit = de.solve(baseline=base, problem=1, mode=2, npop=NPOP, ngen=NGEN)
    full = integrate({y: plan[y] for y in YEARS}, base=plan[2023])
    de.rc.raise_if_invalid(full, base=plan[2023])   # 硬校验，违规即 raise
    return plan, fit / 1e4


def main():
    rows, best = [], None
    for seed in SEEDS:
        plan, fit = solve_once(seed)
        rows.append({'mode': 2, 'seed': seed, '预算': f'{NPOP}×{NGEN}', '总收益/万元': round(fit, 2)})
        print(f'mode2 seed{seed}: {fit:.2f} 万元', flush=True)
        if best is None or fit > best[1]:
            best = (plan, fit, seed)
    df = pd.DataFrame(rows)
    df.to_csv(BASE / 'outputs' / 'p1_mode2_best.csv', index=False,
              encoding='utf-8-sig', float_format='%.2f')

    shutil.copy(PKL, BASE / 'data' / 'de_plan_full_p1_mode2_s0_32x80.pkl')  # 备份旧基准
    with open(PKL, 'wb') as f:
        pickle.dump(best[0], f)
    print(f'\n最优: seed{best[2]} = {best[1]:.2f} 万元 → 已覆盖 {PKL.name}（旧基准已备份）')


if __name__ == '__main__':
    main()
