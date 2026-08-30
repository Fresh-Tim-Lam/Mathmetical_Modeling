# -*- coding: utf-8 -*-
"""方案表示模块：地块为向量的 0-1 选择 + 面积系数，派生决策变量 x 供收益函数核算。

编码（文档 §3.1~§3.4）：一年方案 YearPlan = {物理地块: {季次: [(作物编号, α), ...]}}
- 同季可多种作物（各占面积系数 α，同季合种），Σα ≤ 1；α=0 即不种/休耕；
- 物理地块 54 块：A1..A6 / B1..B14 / C1..C6 单季；
  D1..D8 / E1..E16 / F1..F4 两季（D：一季水稻16 或 蔬菜17..34，二季 {35,36,37} 至多一种；
  E：一季蔬菜17..34 + 二季食用菌38..41；F：两季蔬菜17..34）；
- 水浇地联动 R1~R5（水稻↔蔬菜互斥、双季模式第二季恰好一种）由 check() 校验。

派生（文档 §3.7）：derive(plan) → x = {(单元, 作物编号): 面积}
- 每个选中作物 j 在对应单元 u 上产生记录 x_{u,j} = α·A_u（α=0 不产生记录）；
- x 直接作为 revenue.profit_det / revenue.profit_stoch 的输入（dict 多键天然支持同季合种）。
"""
from collections import defaultdict

import pandas as pd

DATA = r'd:\AAA_Jupyter\BBB_Competition\2025\C\data'

RICE = 16                    # 水稻
BEAN = {1, 2, 3, 4, 5, 17, 18, 19}     # 豆类（三年豆类约束，见约束模块）
WATER_VEG = set(range(17, 35))         # 水浇地第一季蔬菜
WATER_S2 = {35, 36, 37}                # 水浇地第二季（大白菜/白萝卜/红萝卜）
MUSH = set(range(38, 42))              # 食用菌（普通大棚第二季 38~41）


class SchemeModel:
    def __init__(self):
        self._load()

    def _load(self):
        """地块表.csv → 单元（面积/支持集）、物理地块、地块·季次支持集。"""
        d = pd.read_csv(f'{DATA}/地块表.csv', skipinitialspace=True)
        d['地块编号'] = d['地块编号'].astype(str).str.strip()
        self.unit_area, self.unit_support = {}, {}
        for _, r in d.iterrows():
            u = r['地块编号']
            self.unit_area[u] = float(r['面积/亩'])
            self.unit_support[u] = [int(x) for x in str(r['可种植作物编号']).split(';')
                                    if x.strip().isdigit()]
        # 物理地块 = 单元去季次后缀（A1→A1；D1-1→D1）；地块 → 单元列表
        self.plots, self.plot_units = [], defaultdict(list)
        for u in d['地块编号']:
            p = u if u[0] in 'ABC' else u[:-2]
            self.plot_units[p].append(u)
            if p not in self.plots:
                self.plots.append(p)
        self.plot_type = {p: p[0] for p in self.plots}
        # 地块·季次 → 支持作物（与地块表一致；A/B/C 单季）
        self.season_support = {}
        for p in self.plots:
            if self.plot_type[p] in 'ABC':
                self.season_support[p] = {1: self.unit_support[p]}
            else:
                self.season_support[p] = {1: self.unit_support[f'{p}-1'],
                                          2: self.unit_support[f'{p}-2']}

    def unit_of(self, plot, season):
        """物理地块 → 该季次的决策单元（单季露天即地块本身）。"""
        return plot if self.plot_type[plot] in 'ABC' else f'{plot}-{season}'

    def derive(self, plan):
        """YearPlan → x = {(单元, 作物编号): 面积}（面积 = α·A_u，同季合种为多键）。"""
        x = defaultdict(float)
        for plot, seasons in plan.items():
            for s, crops in seasons.items():
                u = self.unit_of(plot, s)
                for j, alpha in crops:
                    if alpha > 0:
                        x[(u, j)] += alpha * self.unit_area[u]
        return dict(x)

    def check(self, plan):
        """编码级校验，返回违规信息列表（未知地块/支持集外/Σα>1/水浇地联动 R1~R5）。"""
        errs = []
        for plot, seasons in plan.items():
            if plot not in self.plot_type:
                errs.append(f'{plot}: 未知地块')
                continue
            for s, crops in seasons.items():
                if s not in self.season_support[plot]:
                    errs.append(f'{plot}·季{s}: 该季次不存在')
                    continue
                sup = set(self.season_support[plot][s])
                sa = sum(a for _, a in crops)
                if sa > 1 + 1e-9:
                    errs.append(f'{plot}·季{s}: Σα={sa:.2f} > 1')
                for j, a in crops:
                    if a > 0 and j not in sup:
                        errs.append(f'{plot}·季{s}: 作物{j} 不在支持集')
            if self.plot_type[plot] == 'D':
                s1 = {j for j, a in seasons.get(1, []) if a > 0}
                s2 = {j for j, a in seasons.get(2, []) if a > 0}
                if s1 & WATER_S2:
                    errs.append(f'{plot}: 第一季含第二季作物（R4）')
                if RICE in s1 and s1 - {RICE}:
                    errs.append(f'{plot}: 水稻与蔬菜互斥（R1）')
                if RICE in s1 and s2:
                    errs.append(f'{plot}: 单季水稻模式第二季应为空（R1）')
                if s2 and not (s1 & WATER_VEG):
                    errs.append(f'{plot}: 第二季必须依附双季蔬菜模式（R1/R2）')
                if s1 & WATER_VEG and not s2:
                    errs.append(f'{plot}: 双季模式第二季应选恰好一种（R2）')
                if s1 & WATER_VEG and s2 and len(s2) > 1:
                    errs.append(f'{plot}: 双季模式第二季至多一种（R3）')
        return errs


if __name__ == '__main__':
    # 集成验证：2023 实际种植 → YearPlan → derive → 收益（应与 revenue 冒烟一致 592.63 万）
    m = SchemeModel()
    plant = pd.read_csv(f'{DATA}/2023种植情况.csv', skipinitialspace=True)
    plan = defaultdict(lambda: defaultdict(list))
    for _, r in plant.iterrows():
        plot = str(r['种植地块']).strip()
        s = 1 if str(r['种植季次']).strip() in ('单季', '第一季') else 2
        u = m.unit_of(plot, s)
        alpha = float(r['种植面积/亩']) / m.unit_area[u]
        plan[plot][s].append((int(r['作物编号']), alpha))
    errs = m.check(plan)
    print(f'2023 方案校验: {"通过" if not errs else errs}')
    x = m.derive(plan)
    from revenue import RevenueModel
    rv = RevenueModel()
    print(f'2023 实际收益: {rv.profit_det(x, 1) / 1e4:.2f} 万元（应 592.63）')
    print(f'物理地块={len(m.plots)}, 单元={len(m.unit_area)}, 派生记录数={len(x)}')
