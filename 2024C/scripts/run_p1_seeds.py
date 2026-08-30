# -*- coding: utf-8 -*-
"""问题1 多种子求解稳定性：modes 1/2 × seeds 1..5（预算 32×40）。
基准解（seed=0，32×80）已存在 data/de_plan_full_p1_mode{1,2}.pkl。
输出 outputs/p1_seed_stability.csv（含基准行），用于论文 §6.3 求解器种子稳健性。
用法：python run_p1_seeds.py（在 scripts/ 下）。
"""
import pickle
import sys
from pathlib import Path

import pandas as pd

BASE = Path(r'd:\AAA_Jupyter\BBB_Competition\2025\C')
sys.path.insert(0, str(BASE / 'src'))

from de_solver_full import FullDE, YEARS, integrate, load_2023_baseline

NPOP, NGEN = 32, 40


def bench_fit(mode):
    """从基准 pkl（seed=0, 32×80）评估总收益，避免硬编码。"""
    with open(BASE / 'data' / f'de_plan_full_p1_mode{mode}.pkl', 'rb') as f:
        plan = pickle.load(f)
    de = FullDE()
    return sum(de._profit(plan, y, 1, mode, 'normal', 16) for y in YEARS) / 1e4


def solve_once(seed, mode):
    base = load_2023_baseline()
    de = FullDE(seed=seed)
    plan, fit = de.solve(baseline=base, problem=1, mode=mode, npop=NPOP, ngen=NGEN)
    full = integrate({y: plan[y] for y in YEARS}, base=plan[2023])
    de.rc.raise_if_invalid(full, base=plan[2023])   # 硬校验，违规即 raise
    return fit / 1e4


def main():
    rows = [{'mode': 1, 'seed': 0, '预算': '32×80(基准)', '总收益/万元': round(bench_fit(1), 2)},
            {'mode': 2, 'seed': 0, '预算': '32×80(基准)', '总收益/万元': round(bench_fit(2), 2)}]
    for mode in (1, 2):
        for seed in (1, 2, 3, 4, 5):
            fit = solve_once(seed, mode)
            rows.append({'mode': mode, 'seed': seed, '预算': f'{NPOP}×{NGEN}',
                         '总收益/万元': round(fit, 2)})
            print(f'mode{mode} seed{seed}: {fit:.2f} 万元', flush=True)
    df = pd.DataFrame(rows)
    out = BASE / 'outputs' / 'p1_seed_stability.csv'
    df.to_csv(out, index=False, encoding='utf-8-sig', float_format='%.2f')
    piv = df.pivot_table(index='mode', columns='seed', values='总收益/万元', aggfunc='first')
    diff = piv.max(axis=1) - piv.min(axis=1)
    base0 = piv[0]
    summ = pd.DataFrame({'极差/万元': diff,
                         '相对seed0极差/%': diff / base0 * 100})
    print('\n汇总：\n', piv.to_string())
    print('\n', summ.to_string())
    print(f'\n已保存 {out}')


if __name__ == '__main__':
    main()
