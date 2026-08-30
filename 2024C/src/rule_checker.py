# -*- coding: utf-8 -*-
"""规则检查器 + 方案整合器（依据《附件注意事项与说明.md》逐条对应重新设计）。

规则编号体系：每条规则直接对应附件注意事项条目（见下方对照表）。
R1（不重茬）与 B1（豆类三年窗口）与 S/W/P 一样是【硬规则】：方案违反任意一条
即被拒绝，不进入收益计算。所有规则统一由 RuleChecker 提供，不另设约束模块。

┌──────┬────────────────────────────┬──────────────────────────────────────────┐
│ 编号  │ 名称                        │ 依据（附件注意事项与说明.md / 赛题正文）       │
├──────┼────────────────────────────┼──────────────────────────────────────────┤
│ S1   │ 地块存在性                   │ 附件1 地块说明：54 个物理地块                │
│ S2   │ 季次合法性                   │ 附件1 地块说明(1)~(3)：A/B/C 仅一季；D/E/F 两季│
│ S3   │ 作物支持集                   │ 附件1 作物说明(1)(3)(5)(7) + 地块表          │
│ S4   │ 面积系数（同季 Σα≤1、α∈[0,1]）│ 赛题"同季可合种不同作物"                      │
│ S5   │ 同季同作物唯一记录            │ 数据规范化（方案表示约定）                    │
│ W1   │ 水浇地·单季水稻模式           │ 作物说明(2)：一季水稻 → 无其他作物、二季为空   │
│ W2   │ 水浇地·双季蔬菜模式           │ 作物说明(3)：一季蔬菜 → 二季恰一种 35/36/37   │
│ W3   │ 水浇地·第二季至多一种         │ 作物说明(3)"只能…中的一种（便于管理）"         │
│ W4   │ 水浇地·第一季不含根菜         │ 作物说明(4)：大白菜/白萝卜/红萝卜只能二季       │
│ W5   │ 水浇地·第二季非独立           │ 作物说明(2)：二季不能脱离一季蔬菜单独存在      │
│ P1   │ 普通大棚·两季组成             │ 作物说明(5)(6)：一季蔬菜 + 二季食用菌          │
│ P2   │ 智慧大棚·两季蔬菜             │ 作物说明(7)：两季蔬菜（不含 35~37）            │
│ R1   │ 跨年·不重茬【硬】             │ 赛题正文：同地块相邻两茬不能连续                │
│ B1   │ 跨年·豆类三年窗口【硬】       │ 赛题正文：2023 起三年内至少一次豆类            │
│ F1   │ 完整性·54 地块齐全            │ 附件1 地块说明：每年每块地都有安排（空=休耕）    │
└──────┴────────────────────────────┴──────────────────────────────────────────┘

工作流约定（检查优先）：
  rc = RuleChecker()
  rep = rc.check_full(full, base=base)      # 收集全部违规，不抛错
  rc.raise_if_invalid(full, base=base)      # 有违规立即抛 ValueError（附全部明细）
任何"先算收益"的入口都应先调用 raise_if_invalid 兜底——通过后才计算收益，
返回的就是纯收益（不含任何罚分）。

软/硬边界说明：本模块是唯一的规则权威（硬）。DE 求解器内部为了引导连续搜索，
还会对"解码器修复后仍残留的少量违规"按 penalty() 扣分，这只是搜索内部的软引导，
不构成对最终方案的判定；最终方案一律以本模块的硬检查为准（0 违规才放行）。

方案表示约定（与 docs/收益计算与方案表示.md 一致）：
  YearPlan = {物理地块: {季次: [(作物编号, α), ...]}}，α>0 才计选中；空 dict = 休耕。
  crop_sets(plan) → {物理地块: {季次: 作物集合}}（重茬/豆类判定的中间形态）。
"""
from collections import defaultdict, Counter
from dataclasses import dataclass, field

from scheme import BEAN, MUSH, RICE, WATER_S2, WATER_VEG, SchemeModel

YEARS = range(2024, 2031)


@dataclass
class Violation:
    rule: str            # 规则编号（S1~S5 / W1~W5 / P1~P2 / R1 / B1 / F1）
    year: int
    plot: str
    season: int          # 0 = 地块级（跨季/跨年）
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
    """统一规则检查器：check_full(full_plan) → Report（结构化违规列表）。
    规则编号与《附件注意事项与说明.md》逐条对应（见模块 docstring 对照表）。"""

    def __init__(self, scheme=None):
        self.scheme = scheme or SchemeModel()

    # ---------- 方案中间形态 / 跨年规则内部判定 ----------
    def crop_sets(self, plan):
        """YearPlan → {物理地块: {季次: 作物集合}}（α>0 才算选中）。"""
        return {p: {s: {j for j, a in crops if a > 0}
                    for s, crops in seasons.items()}
                for p, seasons in plan.items()}

    def replant_violations(self, prev, curr):
        """R1 不重茬：相邻两年违规。prev/curr 为 crop_sets 形式，返回 [(地块, 说明)]。
        时间序列 S(t−1,一季)→S(t−1,二季)→S(t,一季)→S(t,二季)，空季跳过。"""
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
        """B1 豆类三年窗口：滚动窗口 2023~2025 … 2028~2030（6 个），
        每个窗口内每块地至少有一季种豆类。years = {年份: YearPlan}（含 2023 基线），
        返回 [(物理地块, 窗口起点)]。"""
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
        """DE 搜索内部引导罚分（软），仅供求解器比较中间方案使用：
        总罚（元）= λ1·重茬违规次数 + λ2·豆类窗口违规数。
        注意：最终方案判定一律走 raise_if_invalid 硬检查（0 违规），不经过此罚分。"""
        n1 = sum(len(self.replant_violations(self.crop_sets(years[t - 1]),
                                             self.crop_sets(years[t])))
                 for t in range(2024, 2031) if t in years and t - 1 in years)
        n2 = len(self.bean_violations(years))
        return lam1 * n1 + lam2 * n2

    # ---------- 单年检查（结构 S + 水浇地 W + 大棚 P） ----------
    def check_year(self, plan, year=0):
        vs = []
        for plot, seasons in plan.items():
            t = self.scheme.plot_type.get(plot)
            if t is None:                                # S1 地块存在性
                vs.append(Violation('S1', year, plot, 0, '未知地块'))
                continue
            for s, crops in seasons.items():
                if s not in self.scheme.season_support[plot]:   # S2 季次合法性
                    vs.append(Violation('S2', year, plot, s, '该季次不存在'))
                    continue
                sup = set(self.scheme.season_support[plot][s])
                seen, sa = {}, 0.0
                for j, a in crops:
                    sa += a
                    if not (0.0 <= a <= 1.0 + 1e-9):     # S4 面积系数
                        vs.append(Violation('S4', year, plot, s, f'α={a:.3f} 越界'))
                    if a > 0 and j not in sup:           # S3 作物支持集
                        vs.append(Violation('S3', year, plot, s, f'作物{j} 不在支持集'))
                    if a > 0:
                        seen[j] = seen.get(j, 0) + 1
                if sa > 1 + 1e-9:                        # S4 同季 Σα≤1
                    vs.append(Violation('S4', year, plot, s, f'Σα={sa:.3f} > 1'))
                for j, n in seen.items():                # S5 同季同作物唯一
                    if n > 1:
                        vs.append(Violation('S5', year, plot, s, f'作物{j} 重复 {n} 条'))
            # 水浇地联动（地块级，作物说明 2/3/4）
            if t == 'D':
                s1 = {j for j, a in seasons.get(1, []) if a > 0}
                s2 = {j for j, a in seasons.get(2, []) if a > 0}
                if s1 & WATER_S2:                        # W4 第一季不含根菜
                    vs.append(Violation('W4', year, plot, 1, f'第一季含第二季作物 {s1 & WATER_S2}'))
                if RICE in s1 and s1 - {RICE}:           # W1 单季水稻模式
                    vs.append(Violation('W1', year, plot, 1, f'水稻与蔬菜互斥（{s1}）'))
                if RICE in s1 and s2:                    # W1 单季水稻 → 二季空
                    vs.append(Violation('W1', year, plot, 2, '单季水稻模式第二季应为空'))
                if s2 and not (s1 & WATER_VEG):          # W5 第二季非独立
                    vs.append(Violation('W5', year, plot, 2, '第二季必须依附双季蔬菜模式'))
                if s1 & WATER_VEG and not s2:            # W2 双季模式二季恰一种
                    vs.append(Violation('W2', year, plot, 2, '双季模式第二季应选恰好一种'))
                if s1 & WATER_VEG and s2 and len(s2) > 1:  # W3 第二季至多一种
                    vs.append(Violation('W3', year, plot, 2, f'双季模式第二季至多一种（{s2}）'))
            # 大棚模式（地块级，作物说明 5/6/7；支持集已由 S3 保证，显式复述）
            if t == 'E':
                s1 = {j for j, a in seasons.get(1, []) if a > 0}
                s2 = {j for j, a in seasons.get(2, []) if a > 0}
                if s1 & WATER_S2 or s1 & MUSH:           # P1 普通大棚一季蔬菜
                    vs.append(Violation('P1', year, plot, 1, f'普通大棚一季应为蔬菜17~34（{s1}）'))
                if s2 and not s2 <= MUSH:                # P1 普通大棚二季食用菌
                    vs.append(Violation('P1', year, plot, 2, f'普通大棚二季只能食用菌38~41（{s2}）'))
            if t == 'F':
                for s, crops in seasons.items():
                    js = {j for j, a in crops if a > 0}
                    if js & (WATER_S2 | MUSH):           # P2 智慧大棚两季蔬菜
                        vs.append(Violation('P2', year, plot, s, f'智慧大棚只能蔬菜17~34（{js}）'))
        return vs

    # ---------- 全量检查（单年 S/W/P + 跨年 R1/B1 + 完整性 F1） ----------
    def check_full(self, full, base=None):
        """full = {年份: YearPlan}（2023 基线可放 base 参数）。
        返回 Report。base 传入时以 base 为准（2023 是固定输入，不可被方案覆写）。"""
        full = dict(full)
        if base:
            full = {2023: base, **full}
        vs = []
        for y, plan in full.items():
            vs += self.check_year(plan, year=y)
            missing = set(self.scheme.plots) - set(plan)
            if missing:                                  # F1 地块覆盖
                vs.append(Violation('F1', y, str(sorted(missing)[:6]), 0,
                                    f'缺 {len(missing)} 块地（休耕地应显式给空 dict）'))
        # 跨年：重茬 R1（硬规则，相邻两年，时间序列空季跳过）
        years = sorted(y for y in full if y in YEARS)
        for i, y in enumerate(years):
            prev = full[y - 1] if y - 1 in full else None
            if prev is None:
                continue
            for p, msg in self.replant_violations(self.crop_sets(prev),
                                                  self.crop_sets(full[y])):
                vs.append(Violation('R1', y, p, 0, msg))
        # 跨年：豆类三年窗口 B1（硬规则，2023~2025 … 2028~2030）
        for p, w in self.bean_violations(full):
            vs.append(Violation('B1', w, p, 0, f'窗口 {w}~{w+2} 无豆类'))
        return Report(vs)

    # ---------- 检查优先接口（计算收益前调用） ----------
    def validate(self, full, base=None):
        """收集全部违规，不抛错；返回 Report。"""
        return self.check_full(full, base=base)

    def raise_if_invalid(self, full, base=None):
        """有违规立即抛 ValueError（错误信息含全部违规明细），无违规返回 Report。"""
        rep = self.check_full(full, base=base)
        if not rep.ok:
            msg = f'方案违反 {len(rep.violations)} 条规则（已拒绝计算收益）：\n' \
                  + '\n'.join(f'  {v}' for v in rep.violations)
            raise ValueError(msg)
        return rep


# ---------- 方案整合器 ----------
def integrate(plans, base=None, scheme=None):
    """多来源各年 YearPlan → 规范化完整方案（丢给规则检查器前的一步）。

    规范化：
      1) 补全 54 个物理地块键（缺失 = 空 dict，即休耕，F1 合法化）；
      2) 去除 α ≤ 0 的记录；
      3) 同季内同一作物合并 α（消除 S5 重复记录）；
      4) 年份键统一为 int，可选并入 2023 基线。
    返回 {年份: YearPlan}。"""
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
            nyr[p] = ss                       # 空 dict = 明确休耕
        out[int(y)] = nyr
    if base:
        out[2023] = base
    return out


if __name__ == '__main__':
    import pandas as pd
    from pathlib import Path

    DATA = Path(r'd:\AAA_Jupyter\BBB_Competition\2025\C\data')
    rc = RuleChecker()

    # ① 2023 基线应全通过
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

    # ② 每类规则构造违规样例 → 检查器应逐条抓到
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

    # ③ 重茬 + 豆类
    y24 = {'A1': {1: [(6, 1.0)]}}
    y25 = {'A1': {1: [(6, 1.0)]}}
    rep = rc.check_full({2024: y24, 2025: y25}, base=y25)
    print(f'\n③ 重茬/豆类样例 → 规则命中: {sorted(rep.summary())}')

    # ④ 整合器 + raise_if_invalid
    merged = integrate({2024: {'A1': {1: [(6, 0.4), (6, 0.3)]}}}, base=base)
    n_filled = sum(1 for ss in merged[2024].values() if ss)
    print(f'\n④ 整合器: 2024 共 {len(merged[2024])} 块地键（54 补全），'
          f'其中 {n_filled} 块非空，A1 记录={merged[2024]["A1"]}（0.4+0.3 合并）')
    try:
        rc.raise_if_invalid({2024: {'A1': {1: [(16, 1.0)]}}})
    except ValueError as e:
        print('    raise_if_invalid 捕获违规: ' + str(e).splitlines()[0])
