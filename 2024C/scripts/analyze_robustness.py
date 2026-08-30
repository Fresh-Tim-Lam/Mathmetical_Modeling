# -*- coding: utf-8 -*-
"""误差 / 灵敏度 / 鲁棒性分析（问题2/3 口径对齐，全部基于 CRN 同一样本集）。

误差分析（err_）：
- MC 收敛：N 阶梯 [1000,2000,5000,10000,20000,50000] 下两方案相关 MC 期望、SE=std/√N、相对 N=50000 偏差；
- 求积阶数：n_quad ∈ {16,24,40} 下 profit_stoch 期望（问题2 框架，n_quad=16 为求解/报告口径）。

灵敏度分析（sens_）：
- 相关强度 ρ：块内 ρ_dp/ρ_dc/ρ_pc 与替代/互补对幅度各 ×0.5/×1.5，重构 R 后重评估 p3 方案
  （固定方案的一阶局部灵敏度，避免重解；PSD 修正同 corr_matrix）；
- 不确定区间：销量/产量/成本半宽 ×0.5/×1.5，对 CRN 样本做逐列仿射缩放（相关性对仿射不变）。

鲁棒性分析（rob_）：
- 分布假设：p2/p3 方案 × 四种分布（问题2 框架求积）；
- 相关结构误设：R=I（无相关）vs R，两方案期望/std/CVaR5；
- MC 种子稳健性：seed ∈ {0,1,2} 下两方案期望/标准差。

输出：outputs/robustness_*.csv（7 个）。
用法：python analyze_robustness.py（在 scripts/ 下，需先有 de_plan_full_p2/p3_mode1.pkl）。
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r'd:\AAA_Jupyter\BBB_Competition\2025\C')
sys.path.insert(0, str(BASE / 'src'))

from compare_p2_p3 import cvar5
from de_solver_full import YEARS, load_2023_baseline
from revenue import DISTS, RevenueModel
from scheme import SchemeModel

N_EVAL = 20000      # 评估层情景数
N_REF = 50000       # MC 收敛参考（最大 N）
SEED = 0


def make_R(rev, comp_scale=1.0, sub_scale=1.0, rho_dp=0.4, rho_dc=0.3, rho_pc=0.2):
    """参数化相关矩阵（块内 ρ × 替代/互补幅度），PSD 修正同 corr_matrix。"""
    m = len(rev.crops)
    R = np.eye(3 * m)
    for j in range(m):
        d, p, c = 3 * j, 3 * j + 1, 3 * j + 2
        R[d, p] = R[p, d] = rho_dp
        R[d, c] = R[c, d] = rho_dc
        R[p, c] = R[c, p] = rho_pc
    for a, b, rho in rev.COMP_PAIRS:
        R[3 * (a - 1), 3 * (b - 1)] = R[3 * (b - 1), 3 * (a - 1)] = rho * comp_scale
    for a, b, mag in rev._sub_pairs():
        R[3 * (a - 1), 3 * (b - 1)] = R[3 * (b - 1), 3 * (a - 1)] = -mag * sub_scale
    w, V = np.linalg.eigh(R)
    w = np.clip(w, 1e-8, None)
    R2 = (V * w) @ V.T
    dg = np.sqrt(np.diag(R2))
    return R2 / np.outer(dg, dg)


def scale_samples(rev, base, sales_s=1.0, price_s=1.0, cost_s=1.0, yield_s=1.0):
    """CRN 样本逐列仿射缩放（相关性不变）：z' = mid + (z − mid)·s。"""
    plo, phi = rev.price_range_p3()
    mid = {
        'xid': 0.5 * (rev.wave_sales[:, 0] + rev.wave_sales[:, 1]) / 100.0,
        'xip': 0.5 * (plo + phi),
        'xic': np.zeros(len(rev.crops)),
        'xiq': np.zeros(len(rev.crops)),
    }
    s = {'xid': sales_s, 'xip': price_s, 'xic': cost_s, 'xiq': yield_s}
    return {k: mid[k] + (v - mid[k]) * s[k] for k, v in base.items()}


def tot_stats(rev, sm, plan, samples):
    """7 年合计 MC 情景收益（CRN：各年共用 samples）→ (均值, 标准差, CVaR5)。"""
    tot = sum(rev.profit_mc(sm.derive(plan[y]), y, mode=1, samples=samples) for y in YEARS)
    return float(tot.mean()), float(tot.std()), cvar5(tot)


def tot_stoch(rev, sm, plan, dist, n_quad):
    return sum(rev.profit_stoch(sm.derive(plan[y]), y, dist=dist, mode=1, n_quad=n_quad)
               for y in YEARS)


def main():
    rev = RevenueModel()
    sm = SchemeModel()
    base = load_2023_baseline(sm)
    R = rev.corr_matrix()
    with open(BASE / 'data' / 'de_plan_full_p2_mode1.pkl', 'rb') as f:
        p2 = pickle.load(f)
    with open(BASE / 'data' / 'de_plan_full_p3_mode1.pkl', 'rb') as f:
        p3 = pickle.load(f)
    out = BASE / 'outputs'
    out.mkdir(exist_ok=True)
    W = '%.2f'

    # ---------- ① 误差分析：MC 收敛 ----------
    sp_ref = rev.mc_samples(2024, N=N_REF, R=R, seed=SEED)
    m3_ref, _, _ = tot_stats(rev, sm, p3, sp_ref)
    rows = []
    for N in [1000, 2000, 5000, 10000, 20000, 50000]:
        sp = rev.mc_samples(2024, N=N, R=R, seed=SEED)
        m2, s2, _ = tot_stats(rev, sm, p2, sp)
        m3, s3, _ = tot_stats(rev, sm, p3, sp)
        rows.append({'N': N, 'p2期望/万元': m2 / 1e4, 'p3期望/万元': m3 / 1e4,
                     'p3_SE/万元': s3 / 1e4 / np.sqrt(N),
                     'p3偏差vsN=50000/%': (m3 / m3_ref - 1) * 100})
    pd.DataFrame(rows).to_csv(out / 'robustness_err_mc.csv', index=False,
                              encoding='utf-8-sig', float_format=W)

    # ---------- ② 误差分析：求积阶数 ----------
    rows = []
    for nq in (16, 24, 40):
        rows.append({'n_quad': nq,
                     'p2期望/万元': tot_stoch(rev, sm, p2, 'normal', nq) / 1e4,
                     'p3期望/万元': tot_stoch(rev, sm, p3, 'normal', nq) / 1e4})
    pd.DataFrame(rows).to_csv(out / 'robustness_err_quad.csv', index=False,
                              encoding='utf-8-sig', float_format=W)

    # ---------- ③ 灵敏度：相关强度 ρ ----------
    sp0 = rev.mc_samples(2024, N=N_EVAL, R=R, seed=SEED)
    m3_0, _, _ = tot_stats(rev, sm, p3, sp0)
    rows = [{'参数': '基准', '取值': '-', 'p3期望/万元': m3_0 / 1e4, 'Δ/万元': 0.0, 'Δ/%': 0.0}]
    for name, kw in [('ρ_dp', {'rho_dp': 0.2}), ('ρ_dp', {'rho_dp': 0.6}),
                     ('ρ_dc', {'rho_dc': 0.15}), ('ρ_dc', {'rho_dc': 0.45}),
                     ('ρ_pc', {'rho_pc': 0.1}), ('ρ_pc', {'rho_pc': 0.3}),
                     ('替代幅度', {'sub_scale': 0.5}), ('替代幅度', {'sub_scale': 1.5}),
                     ('互补幅度', {'comp_scale': 0.5}), ('互补幅度', {'comp_scale': 1.5})]:
        Rv = make_R(rev, **kw)
        sp = rev.mc_samples(2024, N=N_EVAL, R=Rv, seed=SEED)
        m3, _, _ = tot_stats(rev, sm, p3, sp)
        rows.append({'参数': name, '取值': str(kw), 'p3期望/万元': m3 / 1e4,
                     'Δ/万元': (m3 - m3_0) / 1e4, 'Δ/%': (m3 / m3_0 - 1) * 100})
    pd.DataFrame(rows).to_csv(out / 'robustness_sens_rho.csv', index=False,
                              encoding='utf-8-sig', float_format=W)

    # ---------- ④ 灵敏度：不确定区间缩放（p2/p3 方案并排） ----------
    m2_0, _, _ = tot_stats(rev, sm, p2, sp0)
    rows = [{'因素': '基准', '缩放': 1.0, 'p2期望/万元': m2_0 / 1e4, 'p2Δ/%': 0.0,
             'p3期望/万元': m3_0 / 1e4, 'p3Δ/%': 0.0}]
    for name in ('sales', 'price', 'cost', 'yield'):
        for s in (0.5, 1.5):
            kw = {f'{name}_s': s}
            sp = scale_samples(rev, sp0, **kw)
            m2, _, _ = tot_stats(rev, sm, p2, sp)
            m3, _, _ = tot_stats(rev, sm, p3, sp)
            rows.append({'因素': name, '缩放': s, 'p2期望/万元': m2 / 1e4,
                         'p2Δ/%': (m2 / m2_0 - 1) * 100, 'p3期望/万元': m3 / 1e4,
                         'p3Δ/%': (m3 / m3_0 - 1) * 100})
    pd.DataFrame(rows).to_csv(out / 'robustness_sens_range.csv', index=False,
                              encoding='utf-8-sig', float_format=W)

    # ---------- ⑤ 鲁棒性：分布假设 ----------
    rows = []
    for dist in DISTS:
        rows.append({'分布': dist, 'p2期望/万元': tot_stoch(rev, sm, p2, dist, 16) / 1e4,
                     'p3期望/万元': tot_stoch(rev, sm, p3, dist, 16) / 1e4})
    pd.DataFrame(rows).to_csv(out / 'robustness_rob_dist.csv', index=False,
                              encoding='utf-8-sig', float_format=W)

    # ---------- ⑥ 鲁棒性：相关结构误设（R=I vs R） ----------
    spI = rev.mc_samples(2024, N=N_EVAL, R=None, seed=SEED)
    rows = []
    for label, sp in [('R=I（独立）', spI), ('R（相关）', sp0)]:
        m2, s2, c2 = tot_stats(rev, sm, p2, sp)
        m3, s3, c3 = tot_stats(rev, sm, p3, sp)
        rows.append({'相关假设': label, 'p2期望/万元': m2 / 1e4, 'p2std/万元': s2 / 1e4,
                     'p2CVaR5/万元': c2 / 1e4, 'p3期望/万元': m3 / 1e4, 'p3std/万元': s3 / 1e4,
                     'p3CVaR5/万元': c3 / 1e4})
    pd.DataFrame(rows).to_csv(out / 'robustness_rob_indep.csv', index=False,
                              encoding='utf-8-sig', float_format=W)

    # ---------- ⑦ 鲁棒性：MC 种子 ----------
    rows = []
    for sd in (0, 1, 2):
        sp = rev.mc_samples(2024, N=N_EVAL, R=R, seed=sd)
        m2, s2, _ = tot_stats(rev, sm, p2, sp)
        m3, s3, _ = tot_stats(rev, sm, p3, sp)
        rows.append({'seed': sd, 'p2期望/万元': m2 / 1e4, 'p3期望/万元': m3 / 1e4,
                     'p2std/万元': s2 / 1e4, 'p3std/万元': s3 / 1e4})
    pd.DataFrame(rows).to_csv(out / 'robustness_rob_seed.csv', index=False,
                              encoding='utf-8-sig', float_format=W)

    for f in sorted(out.glob('robustness_*.csv')):
        print(f'== {f.name} ==')
        print(pd.read_csv(f).to_string(index=False))
        print()


if __name__ == '__main__':
    main()
