# -*- coding: utf-8 -*-
"""收益模块：问题1 确定性收益 profit_det（det = deterministic）+ 问题2/3 期望收益 profit_stoch（stoch = stochastic，分布函数积分）。

- 参数唯一源：data/经济参数明细.csv → 档位参数矩阵 T[10,41]=(q,c,p)；
  方案经 to_ts(x) 对齐为同构档位面积矩阵 X_ts[10,41]（收益函数唯一入口）。
- 滞销按作物级：r_j = min(1, D_j/Y_j)；产值系数 ψ = r + κ(1−r)（κ=0/0.5 ↔ mode=1/2）。
- profit_stoch：E[π] = Σ_j PX_j·E[1+ξ_pj]·I_j − C·1.05^τ（逐作物二维 Gauss 求积）。
- 函数分工：profit_det/profit_stoch 对外收益；to_ts/_unit_x 方案对齐；
  _load/_ts_of 数据加载；_quad 求积节点分发；四个分布函数（uniform_dist 等）即 dist 的输入接口。

调用：
    m = RevenueModel()
    m.profit_det(x, mode=1)                                   # 问题1
    mu = m.profit_stoch(x, 2024, dist='normal', mode=1)       # 问题2/3

索引（内部整数）：单元 0..81；作物 crop_idx[j]=j-1（0..40）；档位 ts_idx：A,B,C,D,D-1,D-2,E-1,E-2,F-1,F-2。
分布：uniform / triangular / normal(3σ) / logit_normal（有限区间经 sigmoid 映射到实轴再配正态）。
"""
import re

import numpy as np
import pandas as pd

from data_prep import crop_sales_2023

BASE = r'd:\AAA_Jupyter\BBB_Competition\2025\C'
DOCS = f'{BASE}/docs'
DATA = f'{BASE}/data'

DISTS = ('uniform', 'triangular', 'normal', 'logit_normal')

GROW_SALES = (6, 7)          # 小麦/玉米：预期销量复合增长（g_j = (1+ξ)^τ）


def parse_range(s):
    """波动/价格字符串 → (lo, hi)。'±5%'→(-5,5) '+5%~+10%'→(5,10) '0%'→(0,0)
    '2.50-4.00'→(2.5,4) '-1%~-5%'→(-5,-1)"""
    s = str(s).strip().replace('%', '').replace('+', '')
    if '±' in s:
        v = float(s.replace('±', ''))
        return -v, v
    nums = [float(x) for x in re.findall(r'(?<!\d)-?\d+(?:\.\d+)?', s)]
    if not nums:
        return 0.0, 0.0
    if len(nums) == 1:
        return nums[0], nums[0]
    return min(nums), max(nums) # 保证 lo <= hi


def uniform_dist(lo, hi, n=40):
    """均匀分布 U(lo,hi) 求积节点/权重（权重已含密度 1/(hi−lo)）；lo==hi 退化为单点。"""
    if hi <= lo:
        return np.array([float(lo)]), np.array([1.0])
    x, w = np.polynomial.legendre.leggauss(n)
    mid = (lo + hi) / 2
    x = (hi - lo) / 2 * x + mid
    return x, w / 2          # (hi−lo)/2·w·1/(hi−lo)


def triangular_dist(lo, hi, n=40):
    """三角分布 Tri(lo, mid, hi) 求积节点/权重（权重含密度 f）。"""
    if hi <= lo:
        return np.array([float(lo)]), np.array([1.0])
    x, w = np.polynomial.legendre.leggauss(n)
    mid = (lo + hi) / 2
    x = (hi - lo) / 2 * x + mid
    w = (hi - lo) / 2 * w
    f = np.where(x <= mid, 2 * (x - lo) / ((hi - lo) * (mid - lo)),
                 2 * (hi - x) / ((hi - lo) * (hi - mid)))
    return x, w * f


def normal_dist(lo, hi, n=40):
    """正态（3σ）N(mid, (hi−lo)/6) 求积节点/权重（权重含密度 f，支撑截断到 [lo,hi]）。"""
    if hi <= lo:
        return np.array([float(lo)]), np.array([1.0])
    x, w = np.polynomial.legendre.leggauss(n)
    mid = (lo + hi) / 2
    x = (hi - lo) / 2 * x + mid
    w = (hi - lo) / 2 * w
    s = (hi - lo) / 6
    f = np.exp(-(x - mid) ** 2 / (2 * s ** 2)) / (s * np.sqrt(2 * np.pi))
    return x, w * f


def logit_normal_dist(lo, hi, n=40):
    """logit-正态：η~N(0,1)，ξ = lo + (hi−lo)·sigmoid(η)。
    Gauss-Hermite 节点 z = η·√2（∫e^{-t²}g(t)dt ≈ Σwᵢg(tᵢ)），权重含标准正态核。"""
    if hi <= lo:
        return np.array([float(lo)]), np.array([1.0])
    z, w = np.polynomial.hermite.hermgauss(n)
    x = lo + (hi - lo) / (1 + np.exp(-z * np.sqrt(2.0)))
    return x, w / np.sqrt(np.pi)


QUAD = {'uniform': uniform_dist, 'triangular': triangular_dist,
        'normal': normal_dist, 'logit_normal': logit_normal_dist}


class RevenueModel:
    def __init__(self):
        self._load()

    # ---------- 方案格式处理（收益函数唯一入口，见 4.1 第 4) 条） ----------
    def _unit_x(self, x):
        """dict 方案 {(单元, 作物编号): 面积} → 单元面积矩阵 X_unit[82,41]。"""
        if isinstance(x, np.ndarray):
            return np.asarray(x, dtype=float)
        X = np.zeros((len(self.units), len(self.crops)))
        for (u, j), area in x.items():
            if area > 0:
                iu = u if isinstance(u, int) else self._unit_idx[u]
                X[iu, self.crop_idx[j]] += area
        return X

    def to_ts(self, x):
        """单元面积矩阵 / dict（随地块·季次不种而 vary）→ 档位面积矩阵 X_ts[10,41]。
        同一档位下所有单元的面积累计到一行，与 经济参数明细.csv 格式同构。"""
        X_unit = self._unit_x(x)
        X_ts = np.zeros((len(self.ts_list), len(self.crops)))
        for it in range(len(self.ts_list)):
            X_ts[it] = (X_unit * (self.ts_map == it)).sum(axis=0)
        return X_ts

    # ---------- 数据加载 ----------
    def _load(self):
        # 1) 索引体系
        self.ts_list = ['A', 'B', 'C', 'D', 'D-1', 'D-2', 'E-1', 'E-2', 'F-1', 'F-2']
        self.ts_idx = {t: i for i, t in enumerate(self.ts_list)}
        self.crops = list(range(1, 42))                 # 作物编号 1..41
        self.crop_idx = {j: j - 1 for j in self.crops}  # 0-based

        # 2) 经济参数明细.csv → 档位参数矩阵 T[ts_idx, crop_idx] = (q, c, p)
        d = pd.read_csv(f'{DATA}/经济参数明细.csv', skipinitialspace=True)
        Tq = np.full((len(self.ts_list), len(self.crops)), np.nan)
        Tc = np.full_like(Tq, np.nan)
        Tp = np.full_like(Tq, np.nan)
        for _, r in d.iterrows():
            it = self.ts_idx[str(r['类型-季节编号']).strip()]
            jc = self.crop_idx[int(r['作物编号'])]
            Tq[it, jc], Tc[it, jc], Tp[it, jc] = (float(r['亩产量/斤']),
                                                 float(r['种植成本/(元/亩)']),
                                                 float(r['售价中值/(元/斤)']))
        self.q = np.nan_to_num(Tq)    # 无参数档位 → 0
        self.c = np.nan_to_num(Tc)
        self.p = np.nan_to_num(Tp)

        # 3) 地块表 → 单元、面积、支持集
        plot = pd.read_csv(f'{DATA}/地块表.csv', skipinitialspace=True)
        plot['地块编号'] = plot['地块编号'].astype(str).str.strip()
        self.units = list(plot['地块编号'])             # 单元 index 即列表下标
        self._unit_idx = {u: i for i, u in enumerate(self.units)}
        self.unit_area = np.array(pd.to_numeric(plot['面积/亩']), dtype=float)  # (82,)
        self.support = {}
        for i, r in plot.iterrows():
            self.support[i] = {self.crop_idx[int(x)] for x in str(r['可种植作物编号']).split(';')
                               if x.strip().isdigit()}

        # 4) 单元→档位归属的累计规则：ts_map[unit_idx, crop_idx]（-1 = 不支持）
        self.ts_map = np.full((len(self.units), len(self.crops)), -1, dtype=int)
        for iu in range(len(self.units)):
            for jc in self.support[iu]:
                self.ts_map[iu, jc] = self.ts_idx[self._ts_of(iu, jc)]

        # 5) 销量基准（作物级总量）：2023 无销量数据，按产量=销量近似。
        #    由 src/data_prep.crop_sales_2023() 直接从附件2 加工（Σ 面积×档位亩产量）。
        #    仅作物级可用——2023 年并非每种作物在每个类型·季节都有种植记录，
        #    单元/档位级基准大量缺失（=0 不代表卖不掉），不能作独立滞销基准。
        sales0 = crop_sales_2023()
        self.sales0 = np.array([sales0[j] for j in self.crops])                      # (41,)

        # 6) 作物级波动区间（经济数据.csv 波动规则表，编号为键 → 数组化）
        s = pd.read_csv(f'{DATA}/经济数据.csv', skipinitialspace=True)
        s['作物编号'] = pd.to_numeric(s['作物编号'])
        self.wave_sales = np.array([parse_range(s.loc[s['作物编号'] == j, '预期销售量年变化率'].iloc[0])
                                    for j in self.crops])                          # (41,2)
        self.wave_yield = np.array([parse_range(s.loc[s['作物编号'] == j, '亩产量年变化率'].iloc[0])
                                    for j in self.crops])
        self.wave_price = np.array([parse_range(s.loc[s['作物编号'] == j, '销售价格年变化率'].iloc[0])
                                    for j in self.crops])

    def _ts_of(self, iu, jc):   # ts = type-season（类型-季节）缩写
        """单元 index + 作物 index → 类型-季节编号（ts：type-season 缩写；字母=类型，数字=季节，单季无后缀）。
        特例：Dx-1 上水稻(j=16) → D（水浇地·单季）。智慧大棚·第一季 F-1 为独立档位
        （附件2 说明其参数与普通大棚·第一季相同，已在 经济参数明细.csv 显式补充）。"""
        u = self.units[iu]
        head = u[0]
        if head in 'ABC':
            return head
        if head == 'D':
            if u.endswith('-2'):
                return 'D-2'
            return 'D' if jc == 15 else 'D-1'   # jc=15 ⇔ 作物编号 16（水稻）
        if head == 'E':
            return 'E-1' if u.endswith('-1') else 'E-2'
        return 'F-1' if u.endswith('-1') else 'F-2'   # 智慧大棚

    # ---------- 收益核算（10×41 档位矩阵；滞销按作物级等比例） ----------
    def profit_det(self, x, mode=1):
        """问题1 确定性收益（元），档位面积矩阵 10×41 纯矩阵核算（各因子=1，无波动）：
        ① 作物总产 Y_j = Σ_ts q·X；② 非滞销/总产比 r_j = min(1, D_j/Y_j)（作物级 D_j，各产区同比例售出）；
        ③ 产值系数 ψ(r_j) = r_j + κ(1−r_j)（κ=0/0.5）；④ 收入 = Σ_j G_j·ψ(r_j)（G_j = Σ_ts p·q·X）；⑤ 收益 = 收入 − 成本。
        mode=1 超量滞销；mode=2 超量按 50% 降价。"""
        X = self.to_ts(x)                     # 档位面积矩阵（10×41）
        prod = (self.q * X).sum(axis=0)       # Y_j：作物总产量
        gross = (self.p * self.q * X).sum(axis=0)   # G_j：作物总产值
        cost = float((self.c * X).sum())      # 总成本
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(prod > 0, self.sales0 / prod, 1.0)   # 非滞销/总产比
        r = np.minimum(r, 1.0)
        kappa = 0.5 if mode == 2 else 0.0     # 统一式 ψ(r) = r + κ(1−r)
        rev = float((gross * (r + kappa * (1.0 - r))).sum())
        return rev - cost

    def _quad(self, lo, hi, dist, n=40):
        """分布积分的求积节点/权重（权重已含密度 f）：∫g(ξ)·f(ξ)dξ ≈ Σ w_k·g(x_k)。
        dist：分布函数（callable，返回 (x, w)）或 QUAD 中的名称（'uniform' 等）。"""
        fn = dist if callable(dist) else QUAD[dist]
        return fn(lo, hi, n)

    def profit_stoch(self, x, year, dist='normal', mode=1, n_quad=40):
        """问题2/3 期望收益 = 分布函数积分（确定性求积，非 MC 抽样，见 §2.2(c)）。

        逐作物分解（各作物独立，成本确定性提出积分号）：
          E[π] = Σ_j PX_j·E[1+ξ_pj]·I_j − C·1.05^τ
          PX_j = Σ_ts p·q·X                    基准产值
          I_j  = ∬ (1+ξ_q)·ψ(r_j)·f_q·f_s dξ_q dξ_s
          r_j  = min(1, A_j·g_j(ξ_s)/(1+ξ_q))  非滞销/总产比
          A_j  = D_j/Ȳ_j，Ȳ_j = Σ_ts q·X      基准供需比
          ψ(r) = r + κ(1−r)，κ=0（mode1）/ 0.5（mode2）

        年演化（§2.2(a)，τ = year−2023）：
          f_q = 1+ξ_q（±10%，每年独立）
          g_j = (1+ξ_s)^τ（小麦/玉米 5%~10% 复合）或 1+ξ_s（其余 ±5%）
          f_p = 1（粮食）/ 1.05^τ（蔬菜）/ (1+ξ_p)^τ（食用菌 -1%~-5%）/ 0.95^τ（羊肚菌）
          c   = 1.05^τ（确定性）
        dist：分布函数（(lo,hi,n)→(x,w)，权重含密度）或名称（'uniform'/'triangular'/'normal'/'logit_normal'）
        返回：期望收益（标量）。"""
        u = year - 2023
        X = self.to_ts(x)
        Yb = (self.q * X).sum(axis=0)                   # [41] 基准产量 Ȳ_j
        PX = (self.p * self.q * X).sum(axis=0)          # [41] 基准产值 Σ_ts p·q·X
        C = float((self.c * X).sum()) * (1.05 ** u)     # 成本确定性年增 5%
        with np.errstate(divide='ignore', invalid='ignore'):
            A = np.where(Yb > 0, self.sales0 / Yb, 0.0)     # [41] 基准供需比
        grow = np.zeros(len(self.crops), dtype=bool)
        grow[list(self.crop_idx[j] for j in GROW_SALES)] = True    # 小麦/玉米 复合增长
        kappa = 0.5 if mode == 2 else 0.0
        plo, phi = self.wave_price[:, 0], self.wave_price[:, 1]
        E = 0.0
        for jc in range(len(self.crops)):
            if A[jc] <= 0 or PX[jc] <= 0:
                continue
            # E[1+ξ_pj]：价格因子（粮食/蔬菜/羊肚菌为确定性常数；食用菌为一维积分）
            pl, ph = plo[jc], phi[jc]
            if pl == ph:
                fp = 1.0 if pl == 0 else (1 + pl / 100.0) ** u
            else:
                xp, wp = self._quad(pl / 100.0, ph / 100.0, dist, n_quad)
                fp = float(((1 + xp) ** u * wp).sum())
            # I_j：二维求积（ξ_q × ξ_s），内层 r_j 用 g_j(ξ_s)
            xq, wq = self._quad(self.wave_yield[jc, 0] / 100.0,
                                self.wave_yield[jc, 1] / 100.0, dist, n_quad)
            xs, ws = self._quad(self.wave_sales[jc, 0] / 100.0,
                                self.wave_sales[jc, 1] / 100.0, dist, n_quad)
            gj = (lambda v: (1 + v) ** u) if grow[jc] else (lambda v: 1 + v)
            den = 1.0 + xq                                # (n_q,)
            gb = gj(xs)                                   # (n_s,)
            r = np.minimum(1.0, A[jc] * gb[None, :] / den[:, None])   # (n_q, n_s) 非滞销/总产比
            psi = r + kappa * (1.0 - r)                   # 产值系数统一式
            I = float((den[:, None] * psi * wq[:, None] * ws[None, :]).sum())
            E += PX[jc] * fp * I
        E -= C
        return E

    # ---------- 问题3：可替代/互补性 + 相关蒙特卡洛 ----------

    # 互补需求对（一起购买 → 销量正相关）：家常搭配/冬季炖菜等
    COMP_PAIRS = [
        (20, 21, 0.25),   # 土豆↔西红柿
        (21, 22, 0.30),   # 西红柿↔茄子（家常配菜）
        (24, 31, 0.30),   # 青椒↔辣椒（辣味配菜）
        (29, 30, 0.25),   # 黄瓜↔生菜（凉拌组合）
        (35, 36, 0.30),   # 大白菜↔白萝卜（冬季炖菜）
        (35, 37, 0.30),   # 大白菜↔红萝卜
    ]

    def _sub_pairs(self):
        """可替代需求对（二选一购买 → 销量负相关），避开互补对。"""
        com = {(a, b) for a, b, _ in self.COMP_PAIRS} | {(b, a) for a, b, _ in self.COMP_PAIRS}
        pairs = []

        def group(g, mag):
            g = sorted(set(g))
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    if (g[i], g[j]) not in com:
                        pairs.append((g[i], g[j], mag))

        for b in range(1, 6):                    # 小麦/玉米 ↔ 豆类(1..5) 主粮-蛋白互替
            pairs.append((6, b, 0.20)); pairs.append((7, b, 0.20))
        pairs.append((6, 7, 0.40))               # 小麦↔玉米（主粮互替）
        pairs.append((6, 16, 0.20))              # 小麦↔水稻（主食互替）
        group((8, 9, 10, 15), 0.20)              # 杂粮：谷子/高粱/黍子/大麦
        group((20, 21, 22, 24, 29, 31), 0.25)    # 果菜
        group((23, 27, 28, 30, 32, 33, 34), 0.25)  # 叶菜
        group((17, 18, 19), 0.25)                # 豆类蔬菜：豇豆/刀豆/芸豆
        group((38, 39, 40, 41), 0.20)            # 食用菌
        pairs.append((36, 37, 0.30))             # 白萝卜↔红萝卜
        return pairs

    def corr_matrix(self):
        """问题3 相关矩阵 R（3m×3m，m=41：每作物 {销量 d, 价格 p, 成本 c}）。
        块内 ρ_dp=0.4、ρ_dc=0.3、ρ_pc=0.2；块间 ρ_dd：可替代<0、互补>0（_sub_pairs/COMP_PAIRS）。
        返回 PSD 修正后的相关矩阵（对角线=1），供 Cholesky 采样。"""
        if getattr(self, '_R', None) is not None:
            return self._R
        m = len(self.crops)
        R = np.eye(3 * m)
        for j in range(m):
            d, p, c = 3 * j, 3 * j + 1, 3 * j + 2
            R[d, p] = R[p, d] = 0.40
            R[d, c] = R[c, d] = 0.30
            R[p, c] = R[c, p] = 0.20
        for a, b, rho in self.COMP_PAIRS:
            R[3 * (a - 1), 3 * (b - 1)] = R[3 * (b - 1), 3 * (a - 1)] = rho
        for a, b, mag in self._sub_pairs():
            R[3 * (a - 1), 3 * (b - 1)] = R[3 * (b - 1), 3 * (a - 1)] = -mag
        w, V = np.linalg.eigh(R)                 # PSD 修正（负特征值钳制 + 对角归一）
        w = np.clip(w, 1e-8, None)
        R2 = (V * w) @ V.T
        dg = np.sqrt(np.diag(R2))
        self._R = R2 / np.outer(dg, dg)
        return self._R

    def price_range_p3(self):
        """问题3 价格波动区间（%/100）：粮食±2%（基本稳定）、蔬菜+3~+7%（年增5%左右）、
        食用菌−5~−1%、羊肚菌确定性−5%。"""
        m = len(self.crops)
        lo = np.full(m, -2.0); hi = np.full(m, 2.0)
        lo[16:37] = 3.0; hi[16:37] = 7.0        # 蔬菜 编号17..37（jc 16..36）
        lo[37:40] = -5.0; hi[37:40] = -1.0      # 食用菌 38..40（jc 37..39）
        lo[40] = hi[40] = -5.0                  # 羊肚菌 41（jc 40）确定性
        return lo / 100.0, hi / 100.0

    def mc_samples(self, year, N=2000, R=None, seed=0):
        """第 year 年 N 组相关冲击样本（CRN：固定 seed → 同一年内所有个体共用同一组样本）。
        3m 维变量 (d,p,c) 由 R 相关（R=None 独立），3σ=区间半宽 截断正态、clamp 到区间；
        q 独立 ±10%。返回 dict {xiq, xid, xip, xic} 各 (N, m)。"""
        rng = np.random.default_rng(seed)
        m = len(self.crops)
        dlo, dhi = self.wave_sales[:, 0] / 100.0, self.wave_sales[:, 1] / 100.0
        plo, phi = self.price_range_p3()
        clo = np.full(m, -0.02); chi = np.full(m, 0.02)   # 成本波动 ±2%
        lo = np.concatenate([dlo, plo, clo]); hi = np.concatenate([dhi, phi, chi])
        mu = 0.5 * (lo + hi); sig = (hi - lo) / 6.0
        z0 = rng.standard_normal((N, 3 * m))
        if R is None:
            z = mu + sig * z0
        else:
            S = R * sig[:, None] * sig[None, :]
            w, V = np.linalg.eigh(S)      # S=V diag(w) V^T；w 钳制保半正定 → L = V·√w
            w = np.clip(w, 1e-10, None)
            L = V * np.sqrt(w)
            z = mu + z0 @ L.T
        z = np.clip(z, lo, hi)
        qlo, qhi = -0.10, 0.10
        xiq = np.clip((qhi - qlo) / 6.0 * rng.standard_normal((N, m)), qlo, qhi)
        return {'xiq': xiq, 'xid': z[:, :m], 'xip': z[:, m:2 * m], 'xic': z[:, 2 * m:]}

    def profit_mc(self, x, year, mode=1, N=2000, R=None, seed=0, samples=None):
        """问题3 期望收益（MC 模拟，"通过模拟数据进行求解"）：返回 N 个情景收益 π_n（元），
        期望 mean(π)、风险 std/CVaR 由调用方统计。
        - 销量因子：麦玉 (1+ξ_d)^τ 复合增长、其余 (1+ξ_d)；
        - 价格因子 fp=(1+ξ_p)^τ（羊肚菌确定性 −5%）；成本 c·1.05^τ·(1+ξ_c)；
        - 产量 (1+ξ_q) 独立；滞销统一式 ψ(r)=r+κ(1−r)（κ=0/0.5 ↔ mode=1/2）。
        samples 给定则复用（CRN），否则按 (N, R, seed) 现采。"""
        u = year - 2023
        X = self.to_ts(x)
        Yb = (self.q * X).sum(axis=0)                     # [m] 基准产量
        PX = (self.p * self.q * X).sum(axis=0)            # [m] 基准产值
        Cj = (self.c * X).sum(axis=0)                     # [m] 作物级成本
        with np.errstate(divide='ignore', invalid='ignore'):
            A = np.where(Yb > 0, self.sales0 / Yb, 0.0)
        kappa = 0.5 if mode == 2 else 0.0
        grow = np.zeros(len(self.crops), dtype=bool)
        grow[list(self.crop_idx[j] for j in GROW_SALES)] = True
        if samples is None:
            samples = self.mc_samples(year, N, R, seed)
        xiq, xid, xip, xic = (samples['xiq'], samples['xid'], samples['xip'], samples['xic'])
        fq = 1.0 + xiq                                     # (N,m) 产量因子
        fd = np.where(grow[None, :], (1.0 + xid) ** u, 1.0 + xid)   # 销量因子
        fp = (1.0 + xip) ** u                              # 价格因子
        Dt = self.sales0[None, :] * fd                     # (N,m) 情景销量
        prod = Yb[None, :] * fq                            # (N,m) 情景总产
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.minimum(1.0, np.where(prod > 0, Dt / prod, 1.0))
        psi = r + kappa * (1.0 - r)                        # 产值系数统一式
        gross = PX[None, :] * fp * fq                      # (N,m) 情景产值
        rev = (gross * psi).sum(axis=1)                    # (N,)
        cost = (Cj[None, :] * (1.0 + xic) * (1.05 ** u)).sum(axis=1)
        return rev - cost                                  # (N,) 情景收益


if __name__ == '__main__':
    # 冒烟验证：还原 2023 实际种植 → 2023 收益基线 + 2024 情景期望（四种分布）
    m = RevenueModel()
    d23 = pd.read_excel(f'{DOCS}/附件2.xlsx', sheet_name='2023年的农作物种植情况')
    d23['种植地块'] = d23['种植地块'].ffill()
    x23 = {}
    for _, r in d23.iterrows():
        u = str(r['种植地块']).strip()
        sq = str(r['种植季次']).strip()
        if sq == '单季':
            uu = u if not u.startswith('D') else f'{u}-1'   # 水浇地单季(水稻)归入第一季单元
        else:
            uu = f'{u}-{1 if sq == "第一季" else 2}'
        x23[(uu, int(r['作物编号']))] = float(r['种植面积/亩'])
    X23_ts = m.to_ts(x23)   # 档位面积矩阵（格式处理：累计到档位行）

    for mode in (1, 2):
        print(f'2023 实际收益(mode={mode}): {m.profit_det(x23, mode) / 1e4:.2f} 万元')
    for dist in DISTS:
        mu = m.profit_stoch(x23, 2024, dist=dist, mode=1)
        print(f'2024 期望收益({dist}): {mu / 1e4:.2f} 万元')
    mu = m.profit_stoch(x23, 2024, dist=normal_dist, mode=1)   # 分布函数接口
    print(f'2024 期望收益(normal_dist 函数): {mu / 1e4:.2f} 万元')
    # 档位矩阵对齐检查：X_ts 形状 + 各档位累计面积
    print(f'档位矩阵形状={X23_ts.shape}, 总面积={X23_ts.sum():.1f} 亩')
    for it, ts in enumerate(m.ts_list):
        a = X23_ts[it].sum()
        if a > 0:
            print(f'  {ts}: {a:.1f} 亩')
    print(f'支持集组合数={sum(len(m.support[iu]) for iu in m.support)}')
