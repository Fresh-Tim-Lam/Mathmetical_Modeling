
import re

import numpy as np
import pandas as pd

from data_prep import crop_sales_2023

BASE = r'd:\AAA_Jupyter\BBB_Competition\2025\C'
DOCS = f'{BASE}/docs'
DATA = f'{BASE}/data'

DISTS = ('uniform', 'triangular', 'normal', 'logit_normal')

GROW_SALES = (6, 7)

def parse_range(s):
    s = str(s).strip().replace('%', '').replace('+', '')
    if '±' in s:
        v = float(s.replace('±', ''))
        return -v, v
    nums = [float(x) for x in re.findall(r'(?<!\d)-?\d+(?:\.\d+)?', s)]
    if not nums:
        return 0.0, 0.0
    if len(nums) == 1:
        return nums[0], nums[0]
    return min(nums), max(nums)

def uniform_dist(lo, hi, n=40):
    if hi <= lo:
        return np.array([float(lo)]), np.array([1.0])
    x, w = np.polynomial.legendre.leggauss(n)
    mid = (lo + hi) / 2
    x = (hi - lo) / 2 * x + mid
    return x, w / 2

def triangular_dist(lo, hi, n=40):
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

    def _unit_x(self, x):
        if isinstance(x, np.ndarray):
            return np.asarray(x, dtype=float)
        X = np.zeros((len(self.units), len(self.crops)))
        for (u, j), area in x.items():
            if area > 0:
                iu = u if isinstance(u, int) else self._unit_idx[u]
                X[iu, self.crop_idx[j]] += area
        return X

    def to_ts(self, x):
        X_unit = self._unit_x(x)
        X_ts = np.zeros((len(self.ts_list), len(self.crops)))
        for it in range(len(self.ts_list)):
            X_ts[it] = (X_unit * (self.ts_map == it)).sum(axis=0)
        return X_ts

    def _load(self):

        self.ts_list = ['A', 'B', 'C', 'D', 'D-1', 'D-2', 'E-1', 'E-2', 'F-1', 'F-2']
        self.ts_idx = {t: i for i, t in enumerate(self.ts_list)}
        self.crops = list(range(1, 42))
        self.crop_idx = {j: j - 1 for j in self.crops}

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
        self.q = np.nan_to_num(Tq)
        self.c = np.nan_to_num(Tc)
        self.p = np.nan_to_num(Tp)

        plot = pd.read_csv(f'{DATA}/地块表.csv', skipinitialspace=True)
        plot['地块编号'] = plot['地块编号'].astype(str).str.strip()
        self.units = list(plot['地块编号'])
        self._unit_idx = {u: i for i, u in enumerate(self.units)}
        self.unit_area = np.array(pd.to_numeric(plot['面积/亩']), dtype=float)
        self.support = {}
        for i, r in plot.iterrows():
            self.support[i] = {self.crop_idx[int(x)] for x in str(r['可种植作物编号']).split(';')
                               if x.strip().isdigit()}

        self.ts_map = np.full((len(self.units), len(self.crops)), -1, dtype=int)
        for iu in range(len(self.units)):
            for jc in self.support[iu]:
                self.ts_map[iu, jc] = self.ts_idx[self._ts_of(iu, jc)]

        sales0 = crop_sales_2023()
        self.sales0 = np.array([sales0[j] for j in self.crops])

        s = pd.read_csv(f'{DATA}/经济数据.csv', skipinitialspace=True)
        s['作物编号'] = pd.to_numeric(s['作物编号'])
        self.wave_sales = np.array([parse_range(s.loc[s['作物编号'] == j, '预期销售量年变化率'].iloc[0])
                                    for j in self.crops])
        self.wave_yield = np.array([parse_range(s.loc[s['作物编号'] == j, '亩产量年变化率'].iloc[0])
                                    for j in self.crops])
        self.wave_price = np.array([parse_range(s.loc[s['作物编号'] == j, '销售价格年变化率'].iloc[0])
                                    for j in self.crops])

    def _ts_of(self, iu, jc):
        u = self.units[iu]
        head = u[0]
        if head in 'ABC':
            return head
        if head == 'D':
            if u.endswith('-2'):
                return 'D-2'
            return 'D' if jc == 15 else 'D-1'
        if head == 'E':
            return 'E-1' if u.endswith('-1') else 'E-2'
        return 'F-1' if u.endswith('-1') else 'F-2'

    def profit_det(self, x, mode=1):
        X = self.to_ts(x)
        prod = (self.q * X).sum(axis=0)
        gross = (self.p * self.q * X).sum(axis=0)
        cost = float((self.c * X).sum())
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(prod > 0, self.sales0 / prod, 1.0)
        r = np.minimum(r, 1.0)
        kappa = 0.5 if mode == 2 else 0.0
        rev = float((gross * (r + kappa * (1.0 - r))).sum())
        return rev - cost

    def _quad(self, lo, hi, dist, n=40):
        fn = dist if callable(dist) else QUAD[dist]
        return fn(lo, hi, n)

    def profit_stoch(self, x, year, dist='normal', mode=1, n_quad=40):
        u = year - 2023
        X = self.to_ts(x)
        Yb = (self.q * X).sum(axis=0)
        PX = (self.p * self.q * X).sum(axis=0)
        C = float((self.c * X).sum()) * (1.05 ** u)
        with np.errstate(divide='ignore', invalid='ignore'):
            A = np.where(Yb > 0, self.sales0 / Yb, 0.0)
        grow = np.zeros(len(self.crops), dtype=bool)
        grow[list(self.crop_idx[j] for j in GROW_SALES)] = True
        kappa = 0.5 if mode == 2 else 0.0
        plo, phi = self.wave_price[:, 0], self.wave_price[:, 1]
        E = 0.0
        for jc in range(len(self.crops)):
            if A[jc] <= 0 or PX[jc] <= 0:
                continue

            pl, ph = plo[jc], phi[jc]
            if pl == ph:
                fp = 1.0 if pl == 0 else (1 + pl / 100.0) ** u
            else:
                xp, wp = self._quad(pl / 100.0, ph / 100.0, dist, n_quad)
                fp = float(((1 + xp) ** u * wp).sum())

            xq, wq = self._quad(self.wave_yield[jc, 0] / 100.0,
                                self.wave_yield[jc, 1] / 100.0, dist, n_quad)
            xs, ws = self._quad(self.wave_sales[jc, 0] / 100.0,
                                self.wave_sales[jc, 1] / 100.0, dist, n_quad)
            gj = (lambda v: (1 + v) ** u) if grow[jc] else (lambda v: 1 + v)
            den = 1.0 + xq
            gb = gj(xs)
            r = np.minimum(1.0, A[jc] * gb[None, :] / den[:, None])
            psi = r + kappa * (1.0 - r)
            I = float((den[:, None] * psi * wq[:, None] * ws[None, :]).sum())
            E += PX[jc] * fp * I
        E -= C
        return E

    COMP_PAIRS = [
        (20, 21, 0.25),
        (21, 22, 0.30),
        (24, 31, 0.30),
        (29, 30, 0.25),
        (35, 36, 0.30),
        (35, 37, 0.30),
    ]

    def _sub_pairs(self):
        com = {(a, b) for a, b, _ in self.COMP_PAIRS} | {(b, a) for a, b, _ in self.COMP_PAIRS}
        pairs = []

        def group(g, mag):
            g = sorted(set(g))
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    if (g[i], g[j]) not in com:
                        pairs.append((g[i], g[j], mag))

        for b in range(1, 6):
            pairs.append((6, b, 0.20)); pairs.append((7, b, 0.20))
        pairs.append((6, 7, 0.40))
        pairs.append((6, 16, 0.20))
        group((8, 9, 10, 15), 0.20)
        group((20, 21, 22, 24, 29, 31), 0.25)
        group((23, 27, 28, 30, 32, 33, 34), 0.25)
        group((17, 18, 19), 0.25)
        group((38, 39, 40, 41), 0.20)
        pairs.append((36, 37, 0.30))
        return pairs

    def corr_matrix(self):
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
        w, V = np.linalg.eigh(R)
        w = np.clip(w, 1e-8, None)
        R2 = (V * w) @ V.T
        dg = np.sqrt(np.diag(R2))
        self._R = R2 / np.outer(dg, dg)
        return self._R

    def price_range_p3(self):
        m = len(self.crops)
        lo = np.full(m, -2.0); hi = np.full(m, 2.0)
        lo[16:37] = 3.0; hi[16:37] = 7.0
        lo[37:40] = -5.0; hi[37:40] = -1.0
        lo[40] = hi[40] = -5.0
        return lo / 100.0, hi / 100.0

    def mc_samples(self, year, N=2000, R=None, seed=0):
        rng = np.random.default_rng(seed)
        m = len(self.crops)
        dlo, dhi = self.wave_sales[:, 0] / 100.0, self.wave_sales[:, 1] / 100.0
        plo, phi = self.price_range_p3()
        clo = np.full(m, -0.02); chi = np.full(m, 0.02)
        lo = np.concatenate([dlo, plo, clo]); hi = np.concatenate([dhi, phi, chi])
        mu = 0.5 * (lo + hi); sig = (hi - lo) / 6.0
        z0 = rng.standard_normal((N, 3 * m))
        if R is None:
            z = mu + sig * z0
        else:
            S = R * sig[:, None] * sig[None, :]
            w, V = np.linalg.eigh(S)
            w = np.clip(w, 1e-10, None)
            L = V * np.sqrt(w)
            z = mu + z0 @ L.T
        z = np.clip(z, lo, hi)
        qlo, qhi = -0.10, 0.10
        xiq = np.clip((qhi - qlo) / 6.0 * rng.standard_normal((N, m)), qlo, qhi)
        return {'xiq': xiq, 'xid': z[:, :m], 'xip': z[:, m:2 * m], 'xic': z[:, 2 * m:]}

    def profit_mc(self, x, year, mode=1, N=2000, R=None, seed=0, samples=None):
        u = year - 2023
        X = self.to_ts(x)
        Yb = (self.q * X).sum(axis=0)
        PX = (self.p * self.q * X).sum(axis=0)
        Cj = (self.c * X).sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            A = np.where(Yb > 0, self.sales0 / Yb, 0.0)
        kappa = 0.5 if mode == 2 else 0.0
        grow = np.zeros(len(self.crops), dtype=bool)
        grow[list(self.crop_idx[j] for j in GROW_SALES)] = True
        if samples is None:
            samples = self.mc_samples(year, N, R, seed)
        xiq, xid, xip, xic = (samples['xiq'], samples['xid'], samples['xip'], samples['xic'])
        fq = 1.0 + xiq
        fd = np.where(grow[None, :], (1.0 + xid) ** u, 1.0 + xid)
        fp = (1.0 + xip) ** u
        Dt = self.sales0[None, :] * fd
        prod = Yb[None, :] * fq
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.minimum(1.0, np.where(prod > 0, Dt / prod, 1.0))
        psi = r + kappa * (1.0 - r)
        gross = PX[None, :] * fp * fq
        rev = (gross * psi).sum(axis=1)
        cost = (Cj[None, :] * (1.0 + xic) * (1.05 ** u)).sum(axis=1)
        return rev - cost

if __name__ == '__main__':

    m = RevenueModel()
    d23 = pd.read_excel(f'{DOCS}/附件2.xlsx', sheet_name='2023年的农作物种植情况')
    d23['种植地块'] = d23['种植地块'].ffill()
    x23 = {}
    for _, r in d23.iterrows():
        u = str(r['种植地块']).strip()
        sq = str(r['种植季次']).strip()
        if sq == '单季':
            uu = u if not u.startswith('D') else f'{u}-1'
        else:
            uu = f'{u}-{1 if sq == "第一季" else 2}'
        x23[(uu, int(r['作物编号']))] = float(r['种植面积/亩'])
    X23_ts = m.to_ts(x23)

    for mode in (1, 2):
        print(f'2023 实际收益(mode={mode}): {m.profit_det(x23, mode) / 1e4:.2f} 万元')
    for dist in DISTS:
        mu = m.profit_stoch(x23, 2024, dist=dist, mode=1)
        print(f'2024 期望收益({dist}): {mu / 1e4:.2f} 万元')
    mu = m.profit_stoch(x23, 2024, dist=normal_dist, mode=1)
    print(f'2024 期望收益(normal_dist 函数): {mu / 1e4:.2f} 万元')

    print(f'档位矩阵形状={X23_ts.shape}, 总面积={X23_ts.sum():.1f} 亩')
    for it, ts in enumerate(m.ts_list):
        a = X23_ts[it].sum()
        if a > 0:
            print(f'  {ts}: {a:.1f} 亩')
    print(f'支持集组合数={sum(len(m.support[iu]) for iu in m.support)}')
