
from collections import defaultdict

import pandas as pd

DATA = r'd:\AAA_Jupyter\BBB_Competition\2025\C\data'

RICE = 16
BEAN = {1, 2, 3, 4, 5, 17, 18, 19}
WATER_VEG = set(range(17, 35))
WATER_S2 = {35, 36, 37}
MUSH = set(range(38, 42))

class SchemeModel:
    def __init__(self):
        self._load()

    def _load(self):
        d = pd.read_csv(f'{DATA}/地块表.csv', skipinitialspace=True)
        d['地块编号'] = d['地块编号'].astype(str).str.strip()
        self.unit_area, self.unit_support = {}, {}
        for _, r in d.iterrows():
            u = r['地块编号']
            self.unit_area[u] = float(r['面积/亩'])
            self.unit_support[u] = [int(x) for x in str(r['可种植作物编号']).split(';')
                                    if x.strip().isdigit()]

        self.plots, self.plot_units = [], defaultdict(list)
        for u in d['地块编号']:
            p = u if u[0] in 'ABC' else u[:-2]
            self.plot_units[p].append(u)
            if p not in self.plots:
                self.plots.append(p)
        self.plot_type = {p: p[0] for p in self.plots}

        self.season_support = {}
        for p in self.plots:
            if self.plot_type[p] in 'ABC':
                self.season_support[p] = {1: self.unit_support[p]}
            else:
                self.season_support[p] = {1: self.unit_support[f'{p}-1'],
                                          2: self.unit_support[f'{p}-2']}

    def unit_of(self, plot, season):
        return plot if self.plot_type[plot] in 'ABC' else f'{plot}-{season}'

    def derive(self, plan):
        x = defaultdict(float)
        for plot, seasons in plan.items():
            for s, crops in seasons.items():
                u = self.unit_of(plot, s)
                for j, alpha in crops:
                    if alpha > 0:
                        x[(u, j)] += alpha * self.unit_area[u]
        return dict(x)

    def check(self, plan):
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
