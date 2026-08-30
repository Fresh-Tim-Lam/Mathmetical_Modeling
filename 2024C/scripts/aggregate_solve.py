# -*- coding: utf-8 -*-
"""聚合-解聚 MVP：经济参数只依赖【档位=(地块类型,季次)】→ 同档位单元一视同仁。

两阶段：
  ① 聚合层：LP（scipy.optimize.linprog）在 10 档 × 41 作物上求最优连续面积分配
           （销量作物级共享 → 滞销用 w_j≥Y_j−D_j 线性化，D 联动用面积近似约束）；
  ② 解聚层：把每档每种作物的总面积【整块装箱】到具体地块
           （整块优先 α→1，最后一块才部分；逐地块避开上一茬；D 水稻块 D-2 休耕）。
对比当前 solver（逐地块贪心）的收益与"不分散"程度。
"""
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, r'd:\AAA_Jupyter\BBB_Competition\2025\C\src')

from revenue import RevenueModel
from rule_checker import RuleChecker
from scheme import SchemeModel

sm = SchemeModel()
rev = RevenueModel()
cs = RuleChecker(sm)

# 档位：ts_idx 顺序 = [A, B, C, D, D-1, D-2, E-1, E-2, F-1, F-2]
TS = rev.ts_list                      # 档位名列表
gid = {g: i for i, g in enumerate(TS)}
VEG1 = set(range(16, 34))             # 蔬菜 17..34 (0-based 16..33)
RICE = 15                             # 0-based 水稻16
LUOBO = {34, 35, 36}                  # 0-based 大白菜35/白萝卜36/红萝卜37
BEAN = {0, 1, 2, 3, 4, 16, 17, 18}    # 豆类 1..5,17..19 (0-based)

# ---------- 档位面积 ----------
# ts_map[u, jc] = 该单元·作物档位；A_g = 支持该档位的单元面积和
unit_area = rev.unit_area             # (82,)
ts_map = rev.ts_map                   # (82, 41)
Ag = {}
for g, i in gid.items():
    sup = np.any(ts_map == i, axis=1)  # 该档位支持的单元
    Ag[g] = float(unit_area[sup].sum())
# D 单季水稻 与 D-1 蔬菜共享 Dx-1 单元面积
A_D1 = Ag['D']                        # 水浇地第一季单元总面积（水稻/蔬菜共享）
A_D2 = Ag['D-2'] if 'D-2' in Ag else Ag.get('D2', 0.0)

q = rev.q                             # (10,41) 亩产量
c = rev.c                             # (10,41) 成本
p = rev.p                             # (10,41) 售价
D0 = rev.sales0                       # (41,) 预期销量（斤）
pbar = np.where(q.sum(0) > 0, (p * q).sum(0) / np.where(q.sum(0) > 0, q.sum(0), 1), 0.0)

# ---------- ① 聚合层 LP ----------
nG, nJ = 10, 41
# 变量: X[g,jc] (410) + w_j 滞销量 (41)
nX, nW = nG * nJ, nJ
idx = lambda g, jc: g * nJ + jc

row_idx, col_idx, coeff_ub, bub = [], [], [], []
# 面积约束（D 水稻+蔬菜共享 Dx-1 面积）
for _, gs, cap in [('A', ['A'], Ag['A']), ('B', ['B'], Ag['B']), ('C', ['C'], Ag['C']),
                   ('D', ['D', 'D-1'], A_D1), ('D2', ['D-2'], A_D2),
                   ('E1', ['E-1'], Ag['E-1']), ('E2', ['E-2'], Ag['E-2']),
                   ('F1', ['F-1'], Ag['F-1']), ('F2', ['F-2'], Ag['F-2'])]:
    r = len(bub)
    for g in gs:
        gi = gid[g]
        for jc in range(nJ):
            row_idx.append(r); col_idx.append(idx(gi, jc)); coeff_ub.append(1.0)
    bub.append(cap)
# D 联动近似：萝卜总面积 ≤ 蔬菜第一季总面积
r = len(bub)
for jc in LUOBO:
    row_idx.append(r); col_idx.append(idx(gid['D-2'], jc)); coeff_ub.append(1.0)
for jc in VEG1:
    row_idx.append(r); col_idx.append(idx(gid['D-1'], jc)); coeff_ub.append(-1.0)
bub.append(0.0)
# 销量上限：Y_j − w_j ≤ D_j  →  Σ_g q X − w ≤ D
for jc in range(nJ):
    r = len(bub)
    for gi in gid.values():
        if q[gi, jc] > 0:
            row_idx.append(r); col_idx.append(idx(gi, jc)); coeff_ub.append(q[gi, jc])
    row_idx.append(r); col_idx.append(nX + jc); coeff_ub.append(-1.0)
    bub.append(D0[jc])

# 目标: max Σ (p·q−c)·X − Σ pbar·w   →  min Σ (c−p·q)·X + Σ pbar·w
obj = np.zeros(nX + nW)
for gi in gid.values():
    for jc in range(nJ):
        obj[idx(gi, jc)] = c[gi, jc] - p[gi, jc] * q[gi, jc]
for jc in range(nJ):
    obj[nX + jc] = pbar[jc]

from scipy.optimize import linprog
from scipy.sparse import coo_matrix

n_rows = len(bub)
A_ub = coo_matrix((coeff_ub, (row_idx, col_idx)),
                  shape=(n_rows, nX + nW)).tocsr()
res = linprog(obj, A_ub=A_ub, b_ub=np.array(bub, dtype=float),
              bounds=[(0, None)] * (nX + nW), method='highs')
assert res.status == 0, res.message
Xg = res.x[:nX].reshape(nG, nJ)
W = res.x[nX:]
profit_agg = -res.fun
print('==== ① 聚合层 LP 结果 ====')
print(f'档位面积: A={Ag["A"]:.0f} B={Ag["B"]:.0f} C={Ag["C"]:.0f} D1={A_D1:.0f} D2={A_D2:.0f} '
      f'E1={Ag["E-1"]:.0f} E2={Ag["E-2"]:.0f} F1={Ag["F-1"]:.0f} F2={Ag["F-2"]:.0f} 亩')
print(f'LP 最优收益 = {profit_agg/1e4:.2f} 万元（无重茬/整块约束的理论上界）')
nz = [(TS[gi], jc + 1, Xg[gi, jc]) for gi in range(nG) for jc in range(nJ) if Xg[gi, jc] > 0.5]
print(f'面积>0.5亩的(档位,作物,面积) 共 {len(nz)} 条，前 12 条:')
for t in sorted(nz, key=lambda t: -t[2])[:12]:
    print(f'  {t[0]:4} 作物{t[1]:>2}: {t[2]:8.1f} 亩')
print(f'滞销变量 w 非零: {[(jc+1, round(W[jc],0)) for jc in range(nJ) if W[jc] > 1]}')

# ---------- ② 解聚层：整块装箱（避开上一茬 / D 联动） ----------
def plot_of(unit):
    return unit if unit[0] in 'ABC' else unit[:-2]

# 档位 → 候选单元
g_units = defaultdict(list)
for u in range(82):
    for gi in range(nG):
        if np.any(ts_map[u] == gi):
            g_units[gi].append(u)

def disaggregate():
    """整块装箱 → 7 年 YearPlan（不分散仅在此阶段体现，非优化约束）。
    按"作物面积降序 → 地块"装箱：整块优先（α→1），最后一块才部分；
    逐地块避开相邻茬（上一茬 y-1 完整可见），D 联动（水稻块 D-2 休耕、萝卜只进蔬菜块）。"""
    plan = {}
    for y in range(2024, 2031):
        years = {**plan, 2023: base}
        filled = defaultdict(float)                 # 单元已分配面积
        sched = {}                                  # plot -> {s: [(作物编号, α)]}
        items = [(gi, jc, Xg[gi, jc]) for gi in range(nG) for jc in range(nJ) if Xg[gi, jc] > 0.05]
        for gi, jc, S in sorted(items, key=lambda t: -t[2]):
            s = 1 if TS[gi] in ('D', 'D-1', 'E-1', 'F-1') or TS[gi] in ('A', 'B', 'C') else 2
            for u in g_units[gi]:
                if S <= 1e-6:
                    break
                p_ = plot_of(rev.units[u])
                if jc not in rev.support[u] or s not in sm.season_support[p_]:
                    continue
                # 相邻茬（保守）：上一茬 y-1（完整）∪ 本年已填（sched）
                adj = set()
                for y2 in (y - 1, y):
                    for jj, _ in years.get(y2, {}).get(p_, {}).get(2, []):
                        adj.add(rev.crop_idx[jj])
                for y2 in (y - 1, y + 1):
                    for jj, _ in years.get(y2, {}).get(p_, {}).get(1, []):
                        adj.add(rev.crop_idx[jj])
                for jj, _ in sched.get(p_, {}).get(s, []):
                    adj.add(rev.crop_idx[jj])
                if jc in adj:
                    continue
                # D 联动：水稻与蔬菜第一季互斥（R1）；萝卜只进"第一季已种蔬菜"的块（R2/R6）
                blk = sched.get(p_, {})
                if TS[gi] in ('D', 'D-1'):
                    s1_ = {jj for jj, _ in blk.get(1, [])}
                    if (jc == RICE and (s1_ - {16})) or (jc != RICE and 16 in s1_):
                        continue
                if TS[gi] == 'D-2':
                    s1 = {jj for jj, _ in blk.get(1, [])}
                    if not (s1 & set(range(17, 35))) or s1 == {16}:
                        continue
                cap = float(unit_area[u]) - filled[u]
                if cap <= 1e-6:
                    continue
                a = min(cap, S)
                filled[u] += a
                S -= a
                # YearPlan 存 α（面积系数），不是亩
                sched.setdefault(p_, {}).setdefault(s, []).append((jc + 1, a / float(unit_area[u])))
        plan[y] = {p_: {s: crops for s, crops in ss.items() if crops}
                   for p_, ss in sched.items()}
    return plan

base = None  # 需要在函数外构造 2023 基线
# ---- 构造 2023 基线（复用 dump_structure 逻辑）----
import pandas as pd
plant = pd.read_csv(r'd:\AAA_Jupyter\BBB_Competition\2025\C\data\2023种植情况.csv', skipinitialspace=True)
base = defaultdict(lambda: defaultdict(list))
for _, r in plant.iterrows():
    plot = str(r['种植地块']).strip()
    s = 1 if str(r['种植季次']).strip() in ('单季', '第一季') else 2
    u = sm.unit_of(plot, s)
    base[plot][s].append((int(r['作物编号']), float(r['种植面积/亩']) / sm.unit_area[u]))
base = dict(base)

plan0 = disaggregate()

# ---------- ③ 校验 + 收益 ----------
print('\n==== ② 整块装箱 → 校验 ====')
tot_err = tot_re = tot_bean = 0
for y in range(2024, 2031):
    e = sm.check(plan0[y])
    tot_err += len(e)
    prev_sets = cs.crop_sets(base if y == 2024 else plan0[y - 1])
    tot_re += len(cs.replant_violations(prev_sets, cs.crop_sets(plan0[y])))
tot_bean = len(cs.bean_violations(plan0))
print(f'编码违规 {tot_err}，重茬 {tot_re}，豆类 {tot_bean}')
for y in range(2024, 2031):
    e = sm.check(plan0[y])
    if e:
        print(f'  违规示例 {y}: {e[:4]}')
        break

# 分散度：每年每作物占用地块数
def scatter(plan_y):
    cnt = defaultdict(set)
    for p_, ss in plan_y.items():
        for s, crops in ss.items():
            for j, _ in crops:
                cnt[j].add(p_)
    return np.mean([len(v) for v in cnt.values()]), cnt

# 收益
fit = 0.0
for y in range(2024, 2031):
    x = sm.derive(plan0[y])
    fit += rev.profit_det(x, mode=1) / 1e4
print(f'\n聚合-整块解聚 7 年收益 = {fit:.2f} 万元')
print(f'LP 上界 = {profit_agg/1e4:.2f} 万元/年 → 7 年 {7*profit_agg/1e4:.2f} 万，落地率 = {fit/(7*profit_agg/1e4)*100:.1f}%')

# ---------- ④ 与贪心求解器对比（贪心构造见 FullDE._greedy_year，为 DE 热启动种子） ----------
from de_solver_full import FullDE
sol = FullDE(seed=0)
plan_g, fit_g = sol.solve(baseline=base, problem=1, mode=1)
print('\n==== ④ 对比：聚合-整块 vs 当前求解器（全量权重 DE，均含豆类修复+局部搜索）====')
print(f'DE pipeline    收益 = {fit_g/1e4:.2f} 万元')
print(f'聚合-整块      收益 = {fit:.2f} 万元（差 {(fit_g/1e4-fit):.2f} 万）')
print(f'\n分散度（每作物平均地块数，越小越集中）：')
print(f'  {"年份":<6}{"聚合-整块":<12}{"贪心":<12}')
for y in (2024, 2026, 2028):
    m0, _ = scatter(plan0[y])
    mg, _ = scatter(plan_g[y])
    print(f'  {y:<6}{m0:<12.2f}{mg:<12.2f}')

# ---------- ⑤ 聚合解 + 豆类修复 + 局部搜索（与贪心同管线，公平对比） ----------
plan_f = {2023: base, **plan0}
for y in range(2024, 2031):
    for p in sm.plots:
        plan_f[y].setdefault(p, {})            # 休耕地块补空
sol._repair_beans(plan_f)
sol._polish(plan_f, 1, 1, 'normal', 16, max_iter=6)
fit_f = sum(sol._profit(plan_f, y, 1, 1, 'normal', 16) for y in range(2024, 2031)) / 1e4
errs_f = sum(len(sm.check(plan_f[y])) for y in range(2024, 2031))
re_f = sum(len(cs.replant_violations(cs.crop_sets(plan_f[y - 1]), cs.crop_sets(plan_f[y])))
           for y in range(2024, 2031))
bean_f = len(cs.bean_violations(plan_f))
print('\n==== ⑤ 聚合-整块 + 豆类修复 + 局部搜索（公平对比）====')
print(f'聚合 pipeline 收益 = {fit_f:.2f} 万元，编码 {errs_f}/重茬 {re_f}/豆类 {bean_f}')
print(f'DE   pipeline 收益 = {fit_g/1e4:.2f} 万元')
print(f'差异 = {fit_f - fit_g/1e4:.2f} 万元')
print(f'\n最终分散度（每作物平均地块数）：')
for y in (2024, 2026, 2028):
    mf, _ = scatter(plan_f[y])
    mg, _ = scatter(plan_g[y])
    print(f'  {y:<6}聚合 {mf:<10.2f} 贪心 {mg:<10.2f}')
