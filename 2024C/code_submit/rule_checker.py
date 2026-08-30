
from collections import defaultdict, Counter
from dataclasses import dataclass, field

from scheme import BEAN, MUSH, RICE, WATER_S2, WATER_VEG, SchemeModel

YEARS = range(2024, 2031)

@dataclass
class Violation:
    rule: str
    year: int
    plot: str
    season: int
    detail: str

    def __str__(self):
        s = f'{self.year}' if self.year else '  '
        return f'[{self.rule}] {s}年 {self.plot} 季{self.season}：{self.detail}'

@dataclass
class Report:
    violations: list = field(default_factory=list)

    @property
    def ok(self):
        return not self.violations

    def summary(self):
        return dict(Counter(v.rule for v in self.violations))

    def by_rule(self, rule):
        return [v for v in self.violations if v.rule == rule]

    def __str__(self):
        if self.ok:
            return '全部规则通过 ✓'
        out = [f'违规 {len(self.violations)} 条，按规则：{self.summary()}']
        out += [f'  {v}' for v in self.violations[:50]]
        return '\n'.join(out)

class RuleChecker:

    def __init__(self, scheme=None):
        self.scheme = scheme or SchemeModel()

    def crop_sets(self, plan):
        return {p: {s: {j for j, a in crops if a > 0}
                    for s, crops in seasons.items()}
                for p, seasons in plan.items()}

    def replant_violations(self, prev, curr):
        out = []
        for p in set(prev) | set(curr):
            seq = []
            for d in (prev, curr):
                for s in (1, 2):
                    cs_ = d.get(p, {}).get(s)
                    if cs_:
                        seq.append(cs_)
            for i in range(len(seq) - 1):
                inter = seq[i] & seq[i + 1]
                if inter:
                    out.append((p, f'相邻两茬重叠 {inter}'))
        return out

    def bean_violations(self, years):
        sets = {y: self.crop_sets(p) for y, p in years.items()}
        out = []
        for p in self.scheme.plots:
            for w in range(2023, 2029):
                hit = any(seasons & BEAN
                          for y in (w, w + 1, w + 2)
                          for seasons in sets.get(y, {}).get(p, {}).values())
                if not hit:
                    out.append((p, w))
        return out

    def penalty(self, years, lam1=1e6, lam2=1e6):
        n1 = sum(len(self.replant_violations(self.crop_sets(years[t - 1]),
                                             self.crop_sets(years[t])))
                 for t in range(2024, 2031) if t in years and t - 1 in years)
        n2 = len(self.bean_violations(years))
        return lam1 * n1 + lam2 * n2

    def check_year(self, plan, year=0):
        vs = []
        for plot, seasons in plan.items():
            t = self.scheme.plot_type.get(plot)
            if t is None:
                vs.append(Violation('S1', year, plot, 0, '未知地块'))
                continue
            for s, crops in seasons.items():
                if s not in self.scheme.season_support[plot]:
                    vs.append(Violation('S2', year, plot, s, '该季次不存在'))
                    continue
                sup = set(self.scheme.season_support[plot][s])
                seen, sa = {}, 0.0
                for j, a in crops:
                    sa += a
                    if not (0.0 <= a <= 1.0 + 1e-9):
                        vs.append(Violation('S4', year, plot, s, f'α={a:.3f} 越界'))
                    if a > 0 and j not in sup:
                        vs.append(Violation('S3', year, plot, s, f'作物{j} 不在支持集'))
                    if a > 0:
                        seen[j] = seen.get(j, 0) + 1
                if sa > 1 + 1e-9:
                    vs.append(Violation('S4', year, plot, s, f'Σα={sa:.3f} > 1'))
                for j, n in seen.items():
                    if n > 1:
                        vs.append(Violation('S5', year, plot, s, f'作物{j} 重复 {n} 条'))

            if t == 'D':
                s1 = {j for j, a in seasons.get(1, []) if a > 0}
                s2 = {j for j, a in seasons.get(2, []) if a > 0}
                if s1 & WATER_S2:
                    vs.append(Violation('W4', year, plot, 1, f'第一季含第二季作物 {s1 & WATER_S2}'))
                if RICE in s1 and s1 - {RICE}:
                    vs.append(Violation('W1', year, plot, 1, f'水稻与蔬菜互斥（{s1}）'))
                if RICE in s1 and s2:
                    vs.append(Violation('W1', year, plot, 2, '单季水稻模式第二季应为空'))
                if s2 and not (s1 & WATER_VEG):
                    vs.append(Violation('W5', year, plot, 2, '第二季必须依附双季蔬菜模式'))
                if s1 & WATER_VEG and not s2:
                    vs.append(Violation('W2', year, plot, 2, '双季模式第二季应选恰好一种'))
                if s1 & WATER_VEG and s2 and len(s2) > 1:
                    vs.append(Violation('W3', year, plot, 2, f'双季模式第二季至多一种（{s2}）'))

            if t == 'E':
                s1 = {j for j, a in seasons.get(1, []) if a > 0}
                s2 = {j for j, a in seasons.get(2, []) if a > 0}
                if s1 & WATER_S2 or s1 & MUSH:
                    vs.append(Violation('P1', year, plot, 1, f'普通大棚一季应为蔬菜17~34（{s1}）'))
                if s2 and not s2 <= MUSH:
                    vs.append(Violation('P1', year, plot, 2, f'普通大棚二季只能食用菌38~41（{s2}）'))
            if t == 'F':
                for s, crops in seasons.items():
                    js = {j for j, a in crops if a > 0}
                    if js & (WATER_S2 | MUSH):
                        vs.append(Violation('P2', year, plot, s, f'智慧大棚只能蔬菜17~34（{js}）'))
        return vs

    def check_full(self, full, base=None):
        full = dict(full)
        if base:
            full = {2023: base, **full}
        vs = []
        for y, plan in full.items():
            vs += self.check_year(plan, year=y)
            missing = set(self.scheme.plots) - set(plan)
            if missing:
                vs.append(Violation('F1', y, str(sorted(missing)[:6]), 0,
                                    f'缺 {len(missing)} 块地（休耕地应显式给空 dict）'))

        years = sorted(y for y in full if y in YEARS)
        for i, y in enumerate(years):
            prev = full[y - 1] if y - 1 in full else None
            if prev is None:
                continue
            for p, msg in self.replant_violations(self.crop_sets(prev),
                                                  self.crop_sets(full[y])):
                vs.append(Violation('R1', y, p, 0, msg))

        for p, w in self.bean_violations(full):
            vs.append(Violation('B1', w, p, 0, f'窗口 {w}~{w+2} 无豆类'))
        return Report(vs)

    def validate(self, full, base=None):
        return self.check_full(full, base=base)

    def raise_if_invalid(self, full, base=None):
        rep = self.check_full(full, base=base)
        if not rep.ok:
            msg = f'方案违反 {len(rep.violations)} 条规则（已拒绝计算收益）：\n' \
                  + '\n'.join(f'  {v}' for v in rep.violations)
            raise ValueError(msg)
        return rep

def integrate(plans, base=None, scheme=None):
    scheme = scheme or SchemeModel()
    out = {}
    for y, plan in plans.items():
        nyr = {}
        for p in scheme.plots:
            ss = {}
            for s, crops in plan.get(p, {}).items():
                agg = defaultdict(float)
                for j, a in crops:
                    if a > 0:
                        agg[j] += a
                if agg:
                    ss[s] = sorted((j, a) for j, a in agg.items())
            nyr[p] = ss
        out[int(y)] = nyr
    if base:
        out[2023] = base
    return out

if __name__ == '__main__':
    import pandas as pd
    from pathlib import Path

    DATA = Path(r'd:\AAA_Jupyter\BBB_Competition\2025\C\data')
    rc = RuleChecker()

    plant = pd.read_csv(DATA / '2023种植情况.csv', skipinitialspace=True)
    base = defaultdict(lambda: defaultdict(list))
    for _, r in plant.iterrows():
        plot = str(r['种植地块']).strip()
        s = 1 if str(r['种植季次']).strip() in ('单季', '第一季') else 2
        u = rc.scheme.unit_of(plot, s)
        base[plot][s].append((int(r['作物编号']),
                              float(r['种植面积/亩']) / rc.scheme.unit_area[u]))
    base = dict(base)
    rep = rc.check_year(base)
    print('① 2023 基线 check_year:', '通过' if not rep else rep[:3])

    cases = {
        'S1': ({'ZZ9': {1: [(1, 1.0)]}}, '未知地块'),
        'S2': ({'A1': {2: [(1, 1.0)]}}, '季次不存在'),
        'S3': ({'A1': {1: [(16, 1.0)]}}, '水稻不在露天支持集'),
        'S4': ({'A1': {1: [(1, 0.6), (2, 0.6)]}}, 'Σα=1.2>1'),
        'S5': ({'A1': {1: [(1, 0.5), (1, 0.3)]}}, '同作物重复'),
        'W1': ({'D1': {1: [(16, 0.5), (20, 0.5)]}}, '水稻蔬菜互斥'),
        'W1b': ({'D1': {1: [(16, 1.0)], 2: [(35, 1.0)]}}, '单季水稻第二季应空'),
        'W2': ({'D1': {1: [(20, 1.0)]}}, '蔬菜无第二季'),
        'W3': ({'D1': {1: [(20, 1.0)], 2: [(35, 0.5), (36, 0.5)]}}, '第二季两种'),
        'W4': ({'D1': {1: [(35, 1.0)]}}, '第一季含萝卜'),
        'W5': ({'D1': {2: [(35, 1.0)]}}, '第二季独立'),
    }
    print('\n② 规则样例（应全部抓到）：')
    for tag, (plan, desc) in cases.items():
        rep = rc.check_year(plan)
        hit = [v.rule for v in rep]
        print(f'  {tag:<4} {desc:<16} → 抓到 {hit or "无!"}')

    y24 = {'A1': {1: [(6, 1.0)]}}
    y25 = {'A1': {1: [(6, 1.0)]}}
    rep = rc.check_full({2024: y24, 2025: y25}, base=y25)
    print(f'\n③ 重茬/豆类样例 → 规则命中: {sorted(rep.summary())}')

    merged = integrate({2024: {'A1': {1: [(6, 0.4), (6, 0.3)]}}}, base=base)
    n_filled = sum(1 for ss in merged[2024].values() if ss)
    print(f'\n④ 整合器: 2024 共 {len(merged[2024])} 块地键（54 补全），'
          f'其中 {n_filled} 块非空，A1 记录={merged[2024]["A1"]}（0.4+0.3 合并）')
    try:
        rc.raise_if_invalid({2024: {'A1': {1: [(16, 1.0)]}}})
    except ValueError as e:
        print('    raise_if_invalid 捕获违规: ' + str(e).splitlines()[0])
