# -*- coding: utf-8 -*-
"""全量连续权重编码 + 差分进化求解器（当前唯一求解器，src/ 主入口）。

编码（核心创新，对应建模文档 §3.3"全量分配"路线）：
- 对每个(物理地块, 季次)允许的每一种作物，直接给一个连续权重 w。w>0 表示"选择并分配面积"，
  w=0 表示"不选择"——用连续变量的"有无"代替离散 0-1 选择，无需"作物ID+面积"双基因块。
- 变量数 = Σ(每单元允许作物数) = 82 单元共 1062 个连续实数（对话稿按 D-1 每块 18 种计为
  1054，实际地块表 D-1 为 19 种：水稻16 + 蔬菜17..34），离散变量 0。
- 水浇地隐式模式切换公理（附件1 作物说明(2)）：第一季水稻权重 w16 > 0 → Mode1
  （单季水稻，第二季休耕）；w16 = 0 → Mode2（两季蔬菜，第二季根菜）。模式由解码器裁决，
  不引入 0-1 模式变量。

流水线（2024~2030 逐年级进，固定前 y-1 年）：
  ① 贪心构造（_greedy_year，按有效边际 + 需求约束）+ 可行邻域播种 → 种群热启动；
  ② 单年 DE/rand/1/bin（F=0.7, CR=0.9），贪心选择；
  ③ 闭合豆类窗口惩罚 + 缺豆地块确定性强制种豆（_force_bean）；
  ④ mode2 逐年增量修复豆类窗口，mode1 末期一次性修复；
  ⑤ 末期局部搜索（_polish）+ 豆类修复兜底；
  ⑥ 【检查优先】计算最终收益前，先用 RuleChecker 硬检查完整方案（2023 固定基线），
     违规全部收集并 raise；0 违规通过后才返回【纯收益】（不含任何罚分）。

求解流程三阶段（虚拟收益引导 → 硬校验 → 真实收益）：
  ① 【虚拟收益引导】DE 搜索层的目标函数是"虚拟收益"（见 _fitness）：
     F_t(v) = Π_t(真实收益) − P_t(虚拟罚项)，
     P_t = LAM·(编码违规 + 重茬违规 + 闭合窗口缺豆) + SPARSE·作物数，
     LAM=1e6、SPARSE=50 元/作物·季。罚项是"虚拟的"：只存在于求解器引导层，
     不修改收益建模、不进入真实收益（revenue.py 全程不被触碰）。
  ② 【硬校验】解码器内嵌修复链 + _force_bean/_repair_beans 确定性强制种豆保证搜索中方案合法，
     最终方案经 raise_if_invalid 硬门禁（0 违规才放行，2023 固定基线）。
  ③ 【真实收益】硬校验通过后，solve/fitness 末尾返回纯收益（不含任何罚项），即为最终结果。
  虚拟罚项与 RuleChecker.penalty 同源（同一套 R1/B1 判定），只影响搜索方向，不参与最终评价。

解码修复链（_decode）：
- 负权重裁剪；相邻茬权重置零（真实重茬规则：第一季邻上季第二季、第二季邻当季第一季，
  空季跳过，与 RuleChecker.replant_violations 一致）；
- Σα>1 等比缩放（Σα<1 保留部分种植/休耕，适配滞销情形）；
- D 地块模式修复（W1 水稻互斥/单季清二季、W5 清独立二季、W2 补萝卜避开相邻茬）；
- 塌缩修复（第一季被清空、第二季仍有作物 → 重投影合法第一季，防序列塌缩重茬）；
- 补全 54 地块键（空 dict = 休耕）。

2023 基线保护：_repair_beans 只读 2023 的豆类情况、绝不修改 2023（固定输入）；
y=2025 的"前两年"依据冻结的 base_cs 计算，保证三年豆类窗口判定与外部复核一致。
"""
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from revenue import RevenueModel
from rule_checker import RuleChecker, integrate
from scheme import BEAN, RICE, WATER_S2, WATER_VEG, SchemeModel

YEARS = range(2024, 2031)

VEG = list(range(17, 35))          # 蔬菜 17..34（第一季 / 智慧大棚）
MUSH = list(range(38, 41))         # 食用菌 38..40（羊肚菌 41 另列）
S2 = sorted(WATER_S2)              # 水浇地第二季：大白菜/白萝卜/红萝卜

# 默认 DE 参数
NP = 32
G = 80
F = 0.7
CR = 0.9
LAM = 1e6        # 虚拟罚项系数（编码/重茬/闭合窗口缺豆，违规即"几乎不可接受"）
SPARSE = 50.0    # 虚拟罚项·稀疏项（元/作物·季，引导收缩作物数量）
EPS = 0.005      # 解码保留阈值（贪心需求受限分配含小 α 作物，过高会丢失）


class FullDE:
    """全量连续权重编码（1062 维）的差分进化求解器——唯一求解器入口。

    API：solve(baseline, problem, mode, ...) → (plan, fit)；
    任何"先算收益"的调用（solve/fitness/report）都会先过规则检查器。"""

    def __init__(self, scheme=None, revenue=None, seed=0):
        self.seed = seed
        self.sm = scheme or SchemeModel()
        self.rev = revenue or RevenueModel()
        self.rc = RuleChecker(self.sm)
        self.rng = random.Random(seed)
        self.unit_idx = {u: i for i, u in enumerate(self.rev.units)}
        self.crop_idx = {j: j - 1 for j in range(1, 42)}
        self.margin = self._margin_table()          # (82, 41) 全价边际（元/亩）
        # 权重块表：(物理地块, 季次, 支持作物列表, 起始下标)
        self.blocks, off = [], 0
        for p in self.sm.plots:
            for s in sorted(self.sm.season_support[p]):
                sup = list(self.sm.season_support[p][s])
                self.blocks.append((p, s, sup, off))
                off += len(sup)
        self.D = off                                # 1062

    # ---------- 边际（贪心构造 / 候选排序共用） ----------
    def _margin_table(self):
        M = np.full((len(self.rev.units), 41), -np.inf)
        for iu in range(len(self.rev.units)):
            for jc in range(41):
                ts = self.rev.ts_map[iu, jc]
                if ts >= 0:
                    M[iu, jc] = self.rev.p[ts, jc] * self.rev.q[ts, jc] - self.rev.c[ts, jc]
        return M

    def _mg(self, plot, s, j):
        """地块·季次·作物 的原始边际利润（元/亩，p·q − c）。"""
        if j is None:
            return -np.inf
        u = self.sm.unit_of(plot, s)
        return float(self.margin[self.unit_idx[u], self.crop_idx[j]])

    def _eff_margin(self, plot, s, j, rem, mode=1):
        """地块·季次·作物 j 的当前有效边际收益（元/块）：考虑滞销系数 ψ(r)。
        rem = 需求剩余量（斤）；种后总产 Y_new，r = min(1, D/Y_new)，ψ = r + κ(1−r)；
        mode1 超量边际 −c·A_u，mode2 超量 0.5p·q·A_u − c·A_u。"""
        u = self.sm.unit_of(plot, s)
        jc = self.crop_idx[j]
        ts = self.rev.ts_map[self.unit_idx[u], jc]
        if ts < 0:
            return -np.inf
        q = float(self.rev.q[ts, jc])
        p = float(self.rev.p[ts, jc])
        c = float(self.rev.c[ts, jc])
        A_u = float(self.sm.unit_area[u])
        D = float(self.rev.sales0[jc])
        Y_new = (D - max(rem, 0.0)) + q * A_u
        r = min(1.0, D / Y_new) if Y_new > 0 else 1.0
        kappa = 0.5 if mode == 2 else 0.0
        return (r + kappa * (1.0 - r)) * p * q * A_u - c * A_u

    def _demand_state(self, plan_y):
        """一年方案 → 需求剩余量（斤）dict（作物级，D_j − 已种产量）。"""
        used = {j: 0.0 for j in range(1, 42)}
        for p, seasons in plan_y.items():
            for s, cs_ in seasons.items():
                for j, a in cs_:
                    u = self.sm.unit_of(p, s)
                    jc = self.crop_idx[j]
                    ts = self.rev.ts_map[self.unit_idx[u], jc]
                    used[j] += a * float(self.sm.unit_area[u]) * float(self.rev.q[ts, jc])
        return {j: float(self.rev.sales0[j - 1]) - used[j] for j in range(1, 42)}

    # ---------- ① 贪心构造（DE 热启动种子） ----------
    def _fill(self, p, s, cands, demand, mode=1, exclude=(), single=False, max_crops=4):
        """地块·季次 填充：按全价边际降序逐作物合种，面积系数 α 受需求剩余约束。"""
        u = self.sm.unit_of(p, s)
        A_u = float(self.sm.unit_area[u])
        kappa = 0.5 if mode == 2 else 0.0
        out, tot, remain = [], 0.0, 1.0
        taken = set(exclude)
        while remain > 1e-6 and len(out) < max_crops:
            best = None
            for j in cands:
                if j in taken:
                    continue
                jc = self.crop_idx[j]
                ts = self.rev.ts_map[self.unit_idx[u], jc]
                if ts < 0:
                    continue
                q = float(self.rev.q[ts, jc])
                pj = float(self.rev.p[ts, jc])
                c = float(self.rev.c[ts, jc])
                if q <= 0 or pj <= 0:
                    continue
                D = float(self.rev.sales0[jc])
                rem = demand[j]
                Y_u = q * A_u
                if rem > 1e-6:
                    a = min(remain, rem / Y_u)      # 需求可消化的面积 → 全价
                    r = 1.0
                else:
                    a = remain                      # 需求耗尽：超量
                    r = 0.0
                if a <= 1e-6:
                    continue
                psi = r + kappa * (1.0 - r)
                m = psi * pj * Y_u * a - c * A_u * a
                if m <= 0:
                    continue
                if best is None or m > best[0]:
                    best = (m, j, a)
            if best is None:
                break
            m, j, a = best
            out.append((j, a))
            tot += m
            jc = self.crop_idx[j]
            ts = self.rev.ts_map[self.unit_idx[u], jc]
            demand[j] -= a * float(self.rev.q[ts, jc]) * A_u
            remain -= a
            if single:
                break                               # 水浇地第二季恰好一种
            taken.add(j)
        return out, tot

    def _greedy_year(self, prev_sets, mode=1):
        """一年方案（YearPlan）：按全价边际 + 需求精确分配（合种），避开上一茬。"""
        demand = {j: float(self.rev.sales0[j - 1]) for j in range(1, 42)}
        order = sorted(self.sm.plots,
                       key=lambda p: -max((self._mg(p, s, j) for s in self.sm.season_support[p]
                                           for j in self.sm.season_support[p][s]),
                                          default=-np.inf))
        plan = {}
        for p in order:
            t = self.sm.plot_type[p]
            seasons = {}
            if t in 'ABC':
                cs_, _ = self._fill(p, 1, self.sm.season_support[p][1], demand, mode,
                                    exclude=prev_sets.get(p, {}).get(1, set()))
                if cs_:
                    seasons[1] = cs_
            elif t == 'D':
                # 互斥模式：单季水稻 vs 双季蔬菜，试算取总边际高者；双季必有一季蔬菜
                ex1 = prev_sets.get(p, {}).get(1, set()) | prev_sets.get(p, {}).get(2, set())
                d_r, s_r, m_r = dict(demand), {}, 0.0
                c1, m1 = self._fill(p, 1, [RICE], d_r, mode, exclude=ex1)
                if c1:
                    s_r[1], m_r = c1, m1
                d_v, s_v, m_v = dict(demand), {}, 0.0
                c1, m1 = self._fill(p, 1, VEG, d_v, mode, exclude=ex1)
                c2, m2 = self._fill(p, 2, S2, d_v, mode, single=True,
                                    exclude=prev_sets.get(p, {}).get(2, set()))
                if c1:
                    s_v[1] = c1
                    if c2:
                        s_v[2] = c2
                m_v = m1 + m2 if s_v.get(1) else -np.inf
                if m_v > m_r and s_v.get(1):
                    seasons, demand = s_v, d_v
                elif s_r:
                    seasons, demand = s_r, d_r
            elif t == 'E':
                c1, m1 = self._fill(p, 1, VEG, demand, mode)
                c2, m2 = self._fill(p, 2, MUSH + [41], demand, mode)
                if c1:
                    seasons[1] = c1
                if c2:
                    seasons[2] = c2
            else:                                   # F 智慧大棚：两季蔬菜，季内不相交
                ex1 = prev_sets.get(p, {}).get(2, set())
                c1, m1 = self._fill(p, 1, VEG, demand, mode, exclude=ex1)
                ex2 = {j for j, _ in c1}
                c2, m2 = self._fill(p, 2, VEG, demand, mode, exclude=ex2)
                if c1:
                    seasons[1] = c1
                if c2:
                    seasons[2] = c2
            plan[p] = seasons
        return plan

    # ---------- ② 豆类修复（2023 为固定基线，只读不改） ----------
    def _adjacent(self, plan, p, y, s):
        """(y, s) 在重茬时间序列中的相邻茬作物集（跨年/跨季，保守口径）。"""
        adj = set()
        if s == 1:
            for y2 in (y - 1, y):
                for jj, _ in plan.get(y2, {}).get(p, {}).get(2, []):
                    adj.add(jj)
            for y2 in (y - 1, y + 1):
                for jj, _ in plan.get(y2, {}).get(p, {}).get(1, []):
                    adj.add(jj)
        else:
            for y2 in (y, y + 1):
                for jj, _ in plan.get(y2, {}).get(p, {}).get(1, []):
                    adj.add(jj)
            for y2 in (y - 1, y + 1):
                for jj, _ in plan.get(y2, {}).get(p, {}).get(2, []):
                    adj.add(jj)
        return adj

    def _repair_beans(self, plan):
        """滚动 3 年窗口无豆类的地块 → 窗口内（支持豆类的季、替换不引入重茬）
        中边际损失最小的季换成最优豆类。2023 是固定输入，只参与豆类检测、绝不修改。"""
        for p in self.sm.plots:
            for w in range(2023, 2029):
                def beans(y):
                    return {j for s, cs_ in plan.get(y, {}).get(p, {}).items()
                            for j, a in cs_ if a > 0 and j in BEAN}
                if any(beans(y) for y in (w, w + 1, w + 2)):
                    continue
                cand = []
                for y in (w, w + 1, w + 2):
                    if y == 2023 or y not in plan or p not in plan[y]:
                        continue                    # 2023 固定基线不可修改
                    for s, cs_ in plan[y][p].items():
                        sup = self.sm.season_support[p][s]
                        if not (set(sup) & BEAN):
                            continue
                        j = cs_[0][0] if cs_ else None
                        cand.append((self._mg(p, s, j) if j is not None else 0.0, y, s))
                if not cand:
                    continue
                _, y, s = min(cand)                 # 边际最低的季（损失最小）
                adj = self._adjacent(plan, p, y, s)
                ok = [j for j in self.sm.season_support[p][s]
                      if j in BEAN and j not in adj]
                if not ok:
                    continue
                nb = max(ok, key=lambda j: self._mg(p, s, j))
                # D 地块把第一季（原单季水稻）换成豆类蔬菜后，必须补上第二季
                if self.sm.plot_type[p] == 'D' and s == 1 and not plan[y][p].get(2):
                    adj2 = self._adjacent(plan, p, y, 2) | {nb}
                    ok2 = [j for j in S2 if j not in adj2]
                    if not ok2:
                        continue
                    plan[y][p][2] = [(max(ok2, key=lambda j: self._mg(p, 2, j)), 1.0)]
                plan[y][p][s] = [(nb, 1.0)]

    # ---------- 编码 / 解码 ----------
    def _encode(self, plan_y):
        """YearPlan → D 维向量（选中作物的 α 直接写入对应权重槽，未选为 0）。"""
        vec = np.zeros(self.D)
        for p, s, sup, off in self.blocks:
            for j, a in plan_y.get(p, {}).get(s, []):
                if j in sup:
                    vec[off + sup.index(j)] = a
        return vec

    def _decode(self, vec, y, prev_sets):
        """向量 → YearPlan（含全部解码修复）。"""
        plan = defaultdict(lambda: defaultdict(list))
        for p, s, sup, off in self.blocks:
            t = self.sm.plot_type[p]
            # 相邻茬（真实规则：空季跳过）：
            # 第一季邻 prev 第二季（prev 第二季为空时才邻 prev 第一季）；
            # 第二季邻当季第一季（当季第一季为空时由塌缩修复保证非空）
            adj = set()
            if s == 1:
                ps2 = set(prev_sets.get(p, {}).get(2, []))
                ps1 = set(prev_sets.get(p, {}).get(1, []))
                adj = ps2 | (ps1 if not ps2 else set())
            else:
                adj = {j for j, _ in plan[p].get(1, [])}
            w = {j: max(0.0, vec[off + i]) for i, j in enumerate(sup)}
            if t == 'D' and s == 1:                 # 水浇地隐式模式切换公理
                if RICE not in adj and w.get(RICE, 0.0) > 0.02:   # Mode1 单季水稻
                    plan[p][1].append((RICE, 1.0))
                    continue
                w.pop(RICE, None)                   # Mode2：水稻权重置零
            if adj:
                for j in adj:
                    w.pop(j, None)
            total = sum(w.values())
            if total > 1.0:                         # Σα>1 等比缩放；Σα<1 部分种植/休耕
                k = 1.0 / total
                w = {j: a * k for j, a in w.items()}
            for j, a in w.items():
                if a >= EPS:
                    plan[p][s].append((j, a))
        # D 地块模式修复（W1 水稻互斥 / W1 单季清二季 / W5 清独立二季 / W2 补萝卜）
        for p, seasons in list(plan.items()):
            if self.sm.plot_type[p] != 'D':
                continue
            s1 = {j for j, a in seasons.get(1, []) if a > 0}
            s2 = {j for j, a in seasons.get(2, []) if a > 0}
            if RICE in s1 and s1 - {RICE}:          # 水稻与蔬菜互斥 → 只留水稻
                seasons[1] = [(RICE, next(a for j, a in seasons[1] if j == RICE))]
                s1 = {RICE}
            if RICE in s1 and s2:                   # 单季水稻 → 第二季清空
                del seasons[2]; s2 = set()
            if s2 and not (s1 & WATER_VEG):         # 第二季不能独立 → 清空
                del seasons[2]; s2 = set()
            if (s1 & WATER_VEG) and not s2:         # 蔬菜无第二季 → 补最优萝卜
                adj = set(s1)
                ok = [j for j in WATER_S2 if j not in adj]
                if ok:
                    seasons[2] = [(max(ok, key=lambda j: self._mg(p, 2, j)), 1.0)]
        # 塌缩修复：第一季被清空、第二季仍有作物（E/F）→ 重投影合法第一季
        for p, seasons in list(plan.items()):
            if self.sm.plot_type[p] not in ('E', 'F'):
                continue
            s2 = seasons.get(2, [])
            if s2 and not seasons.get(1):
                ps2 = set(prev_sets.get(p, {}).get(2, []))
                ps1 = set(prev_sets.get(p, {}).get(1, []))
                sup = list(self.sm.season_support[p][1])
                adj = set(ps2) | {j for j, _ in s2} | (ps1 if not ps2 else set())
                ok = [j for j in sup if j not in adj]
                if ok:
                    seasons[1] = [(max(ok, key=lambda j: self._mg(p, 1, j)), 1.0)]
                else:
                    del seasons[2]
        out = {p: dict(s) for p, s in plan.items()}
        for p in self.sm.plots:                     # 补齐 54 块（空 dict = 休耕）
            out.setdefault(p, {})
        return out

    # ---------- 适应度（DE 内部：软检查 = 罚分） ----------
    def _n_crops(self, plan_y):
        return sum(1 for _, seas in plan_y.items()
                   for _, cs_ in seas.items() for _, a in cs_ if a > 0)

    @staticmethod
    def _bean_in(crop_sets_p):
        """{季: 作物集} 中是否有豆类。"""
        return any(s & BEAN for s in crop_sets_p.values())

    def _bean_close(self, plan_y, y, prev_sets, prev2_sets):
        """本年 y 闭合的豆类窗口 (y-2,y-1,y)：若 y-2/y-1 均无豆，则 y 必须有豆。
        返回缺豆地块数（y<2025 无窗口闭合 → 0）。"""
        if y < 2025:
            return 0
        n = 0
        for p in self.sm.plots:
            if self._bean_in(prev2_sets.get(p, {})) or self._bean_in(prev_sets.get(p, {})):
                continue
            if not self._bean_in(self.rc.crop_sets({p: plan_y.get(p, {})}).get(p, {})):
                n += 1
        return n

    def _virtual_penalty(self, plan_y, y, prev_sets, prev2_sets):
        """虚拟罚项 P_t(v)：仅供求解器引导层使用，不进入真实收益。

        P_t = LAM·(n_enc + n_re + n_bean) + SPARSE·n_crops
        - n_enc  编码违规数（RuleChecker.check_year）
        - n_re   重茬违规数（RuleChecker.replant_violations，真实相邻茬规则）
        - n_bean 本年闭合豆类窗口缺豆地块数（_bean_close）
        - n_crops 作物·季数（稀疏引导）
        LAM=1e6（违规即"几乎不可接受"）、SPARSE=50 元/作物·季。
        它是虚拟量：只参与 DE 个体排序，不修改收益建模、不计入最终收益。
        """
        n_enc = len(self.rc.check_year(plan_y, y))
        n_re = len(self.rc.replant_violations(prev_sets, self.rc.crop_sets(plan_y)))
        n_bean = self._bean_close(plan_y, y, prev_sets, prev2_sets)
        return LAM * (n_enc + n_re + n_bean) + SPARSE * self._n_crops(plan_y)

    def _fitness(self, plan_y, y, prev_sets, prev2_sets, problem, mode, dist, n_quad):
        """DE 搜索层目标 = 【虚拟收益】= 真实收益 − 虚拟罚项：

        F_t(v) = Π_t( decode(v) ) − P_t(v)

        其中 Π_t 为当年真实收益（问题1 确定性 / 问题2 期望，revenue.py，不改动），
        P_t 为虚拟罚项（_virtual_penalty，含编码/重茬/闭合窗口缺豆/稀疏四类）。
        该目标只决定 DE 个体优劣与搜索方向（求解器引导层）；
        最终结果 = 硬校验通过后的真实收益（solve/fitness 末尾，纯收益、无罚项）。"""
        x = self.sm.derive(plan_y)
        profit = (self.rev.profit_det(x, mode) if problem == 1
                  else self.rev.profit_stoch(x, y, dist, mode, n_quad))
        return profit - self._virtual_penalty(plan_y, y, prev_sets, prev2_sets)

    # ---------- 单年 DE ----------
    def _need_bean_plots(self, y, prev_sets, prev2_sets):
        """本年 y 闭合窗口 (y-2,y-1,y) 中前两年均无豆、须本年种豆的地块列表。"""
        if y < 2025:
            return []
        return [p for p in self.sm.plots
                if not self._bean_in(prev2_sets.get(p, {}))
                and not self._bean_in(prev_sets.get(p, {}))]

    def _force_bean(self, vec, p, prev_sets):
        """向量中给地块 p 的第一季置入一个不重茬的最优豆类作物（返回是否成功）。
        D 地块同时清零水稻权重（Mode2 双季蔬菜才能种豆）。"""
        for pp, s, sup, off in self.blocks:
            if pp != p or s != 1:
                continue
            if not any(j in BEAN for j in sup):
                return False
            ps2 = set(prev_sets.get(p, {}).get(2, []))
            ps1 = set(prev_sets.get(p, {}).get(1, []))
            adj = ps2 | (ps1 if not ps2 else set())
            ok = [j for j in sup if j in BEAN and j not in adj]
            if not ok:
                return False
            j = max(ok, key=lambda j: self._mg(p, 1, j))
            vec[off + sup.index(j)] = 1.0
            if self.sm.plot_type[p] == 'D':
                rice_i = sup.index(RICE) if RICE in sup else -1
                if rice_i >= 0:
                    vec[off + rice_i] = 0.0         # 水稻权重清零 → Mode2
            return True
        return False

    def _de_year(self, vec0, y, prev_sets, prev2_sets, problem, mode, dist, n_quad,
                 npop=NP, ngen=G, F_=F, CR_=CR):
        rng = np.random.default_rng(self.seed * 1000 + y)
        D = self.D
        need_bean = self._need_bean_plots(y, prev_sets, prev2_sets)
        pop = np.zeros((npop, D))
        pop[0] = vec0                               # 热启动：贪心全量方案
        for i in range(1, npop):                    # 可行邻域播种：α 抖动 + 随机槽注入
            v = vec0.copy()
            nz = np.nonzero(vec0)[0]
            if len(nz):
                v[nz] = np.clip(v[nz] + rng.uniform(-0.15, 0.15, len(nz)), 0.0, 1.0)
            for _ in range(int(rng.integers(1, 4))):
                b = int(rng.integers(len(self.blocks)))
                _, _, sup, off = self.blocks[b]
                v[off + int(rng.integers(len(sup)))] = rng.uniform(0.2, 1.0)
            pop[i] = np.clip(v, 0.0, 1.0)
        for i in range(npop):                       # 闭合窗口缺豆地块强制种豆
            for p in need_bean:
                self._force_bean(pop[i], p, prev_sets)
        fits = [self._fitness(self._decode(pop[i], y, prev_sets),
                              y, prev_sets, prev2_sets, problem, mode, dist, n_quad)
                for i in range(npop)]
        for _ in range(ngen):                       # DE/rand/1/bin
            for i in range(npop):
                a, b, c = rng.choice([j for j in range(npop) if j != i], 3, replace=False)
                v = pop[a] + F_ * (pop[b] - pop[c])
                trial = pop[i].copy()
                mask = rng.random(D) < CR_
                mask[rng.integers(D)] = True
                trial[mask] = v[mask]
                trial = np.clip(trial, 0.0, 1.0)
                ft = self._fitness(self._decode(trial, y, prev_sets),
                                   y, prev_sets, prev2_sets, problem, mode, dist, n_quad)
                if ft > fits[i]:
                    pop[i] = trial
                    fits[i] = ft
        bi = int(np.argmax(fits))
        return pop[bi], fits[bi]

    # ---------- ③ 局部搜索 ----------
    def _candidates(self, p, prev_sets, demand, mode=1):
        """地块 p 的候选方案（局部搜索用）：有效边际 top-N + 水浇地模式 + 休耕。"""
        t = self.sm.plot_type[p]

        def top(s, cands, n=3, exclude=()):
            return sorted([j for j in cands if j not in exclude],
                          key=lambda j: -self._eff_margin(p, s, j, demand[j], mode))[:n]

        outs = [{}]                                 # 休耕
        if t in 'ABC':
            ex = prev_sets.get(1, set())
            for j in top(1, self.sm.season_support[p][1], exclude=ex):
                outs.append({1: [(j, 1.0)]})
        elif t == 'D':
            outs.append({1: [(RICE, 1.0)]})
            for s1v in top(1, VEG, 3):
                for s2v in top(2, S2, 2):
                    outs.append({1: [(s1v, 1.0)], 2: [(s2v, 1.0)]})
        elif t == 'E':
            for s1 in top(1, VEG, 3):
                for s2 in top(2, MUSH + [41], 3):
                    outs.append({1: [(s1, 1.0)], 2: [(s2, 1.0)]})
        else:                                       # F：两季蔬菜，季内不相交
            ex = prev_sets.get(2, set())
            for s1 in top(1, VEG, 3, exclude=ex):
                for s2 in top(2, VEG, 3, exclude={s1}):
                    outs.append({1: [(s1, 1.0)], 2: [(s2, 1.0)]})
        return outs

    def _polish(self, plan, problem, mode, dist, n_quad, max_iter=6):
        """局部搜索：逐(年, 地块)试候选，接受【虚拟收益】提高者（真实收益总和 − 虚拟罚项），
        轮间收敛即停。接受标准与 DE 搜索层一致；最终仍以硬校验后的真实收益为准。"""
        for _ in range(max_iter):
            changed = False
            profs = {yy: self._profit(plan, yy, problem, mode, dist, n_quad) for yy in YEARS}
            for y in YEARS:
                prev_sets = self.rc.crop_sets(plan[y - 1])
                demand = self._demand_state(plan[y])
                other = sum(v for k, v in profs.items() if k != y)
                for p in self.sm.plots:
                    cur = plan[y][p]
                    base = other + profs[y] - self.rc.penalty(plan)
                    best_trial, best_f = cur, base
                    for trial in self._candidates(p, prev_sets.get(p, {}), demand, mode):
                        if trial == cur:
                            continue
                        plan[y][p] = trial
                        fy = self._profit(plan, y, problem, mode, dist, n_quad)
                        f = other + fy - self.rc.penalty(plan)
                        if f > best_f:
                            best_f, best_trial = f, trial
                        plan[y][p] = cur
                    if best_trial is not cur:
                        plan[y][p] = best_trial
                        profs[y] = self._profit(plan, y, problem, mode, dist, n_quad)
                        changed = True
            if not changed:
                break
        return plan

    # ---------- 收益 ----------
    def _profit(self, plan, y, problem, mode, dist, n_quad):
        x = self.sm.derive(plan[y])
        if problem == 1:
            return self.rev.profit_det(x, mode)
        return self.rev.profit_stoch(x, y, dist=dist, mode=mode, n_quad=n_quad)

    def fitness(self, plan, problem=1, mode=1, dist='normal', n_quad=40, base=None):
        """检查优先的适应度：方案先过规则检查器硬门禁（违规收集并 raise），
        通过后返回【纯收益】（0 违规，无需扣罚分）。"""
        self.rc.raise_if_invalid(plan, base=base)
        return sum(self._profit(plan, y, problem, mode, dist, n_quad) for y in YEARS)

    def report(self, plan, mode=1):
        """输出年度收益、约束违规与利用面积（调用前需已通过规则检查）。"""
        for y in YEARS:
            x = self.sm.derive(plan[y])
            area = sum(self.sm.unit_area[u] for u in {u for u, _ in x})
            det = self.rev.profit_det(x, mode) / 1e4
            stoch = self.rev.profit_stoch(x, y, dist='normal', mode=mode) / 1e4
            print(f'{y}: 确定性 {det:.2f} 万  期望 {stoch:.2f} 万  面积 {area:.1f} 亩')
        n1 = sum(len(self.rc.replant_violations(self.rc.crop_sets(plan[t - 1]),
                                                self.rc.crop_sets(plan[t])))
                 for t in range(2024, 2031))
        n2 = len(self.rc.bean_violations(plan))
        print(f'重茬违规={n1}, 豆类窗口违规={n2}')

    # ---------- 主流程 ----------
    def solve(self, baseline=None, problem=1, mode=1, dist='normal',
              n_quad=40, seed=None, npop=NP, ngen=G, verbose=True):
        """逐年级进 DE 求解 2024~2030。返回 (plan, fit)。
        最终收益计算前先做规则检查（2023 固定基线），违规收集并 raise。"""
        if seed is not None:
            self.seed = seed
            self.rng = random.Random(self.seed)
        plan = {2023: baseline or {}}
        base_cs = self.rc.crop_sets(plan[2023])     # 冻结 2023（p2 与复核口径一致）
        greedy = {}
        prev = base_cs
        for y in YEARS:                             # 贪心逐年构造（热启动种子）
            greedy[y] = self._greedy_year(prev, mode)
            prev = self.rc.crop_sets(greedy[y])
        prev = base_cs
        for y in YEARS:
            vec0 = self._encode(greedy[y])
            p2 = base_cs if y - 2 == 2023 else (self.rc.crop_sets(plan[y - 2])
                                                if y - 2 >= 2024 else {})
            best, f = self._de_year(vec0, y, prev, p2, problem, mode, dist, n_quad,
                                    npop=npop, ngen=ngen)
            plan[y] = self._decode(best, y, prev)
            prev = self.rc.crop_sets(plan[y])
            if mode == 2:                           # mode2：逐年增量修复豆类窗口
                self._repair_beans(plan)
                prev = self.rc.crop_sets(plan[y])
            if verbose:
                print(f'  {y}: DE 最优适应度 = {f/1e4:.2f} 万元')
        self._repair_beans(plan)
        self._polish(plan, problem, mode, dist, n_quad, max_iter=6)
        self._repair_beans(plan)
        # 【检查优先】收益计算前验证完整方案（2023 固定基线），违规收集并报错；
        # 0 违规通过后返回纯收益（硬约束，不含罚分）
        full = integrate({y: plan[y] for y in YEARS}, base=plan[2023])
        self.rc.raise_if_invalid(full, base=plan[2023])
        fit = sum(self._profit(plan, y, problem, mode, dist, n_quad) for y in YEARS)
        return plan, fit


def load_2023_baseline(scheme=None):
    """从 data/2023种植情况.csv 构造 2023 基线 YearPlan（固定输入）。"""
    sm = scheme or SchemeModel()
    plant = pd.read_csv(Path(__file__).resolve().parent.parent / 'data' / '2023种植情况.csv',
                        skipinitialspace=True)
    base = defaultdict(lambda: defaultdict(list))
    for _, r in plant.iterrows():
        plot = str(r['种植地块']).strip()
        s = 1 if str(r['种植季次']).strip() in ('单季', '第一季') else 2
        u = sm.unit_of(plot, s)
        base[plot][s].append((int(r['作物编号']),
                              float(r['种植面积/亩']) / sm.unit_area[u]))
    return dict(base)


if __name__ == '__main__':
    import sys
    DATA = Path(__file__).resolve().parent.parent / 'data'
    npop = int(sys.argv[1]) if len(sys.argv) > 1 else NP
    ngen = int(sys.argv[2]) if len(sys.argv) > 2 else G
    mode = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    base = load_2023_baseline()

    print(f'全量分配编码变量数 D = {FullDE().D}')
    de = FullDE(seed=0)
    plan, fit = de.solve(baseline=base, problem=1, mode=mode, npop=npop, ngen=ngen)
    print(f'\nFullDE 问题1 mode{mode} 总适应度 = {fit/1e4:.2f} 万元')
    de.report(plan, mode=mode)
    # 检查优先兜底：收益已算，但方案必须通过检查器（违规即 raise）
    de.rc.raise_if_invalid(integrate({y: plan[y] for y in YEARS}, base=base), base=base)
    print(f'规则检查（{npop}×{ngen}）：全部通过 ✓，方案已保存')
    pickle.dump(plan, open(DATA / f'de_plan_full_mode{mode}.pkl', 'wb'))
