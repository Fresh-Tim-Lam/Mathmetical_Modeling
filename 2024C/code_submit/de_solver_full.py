
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

VEG = list(range(17, 35))
MUSH = list(range(38, 41))
S2 = sorted(WATER_S2)

NP = 32
G = 80
F = 0.7
CR = 0.9
LAM = 1e6
SPARSE = 50.0
EPS = 0.005

class FullDE:

    def __init__(self, scheme=None, revenue=None, seed=0):
        self.seed = seed
        self.sm = scheme or SchemeModel()
        self.rev = revenue or RevenueModel()
        self.rc = RuleChecker(self.sm)
        self.rng = random.Random(seed)
        self.unit_idx = {u: i for i, u in enumerate(self.rev.units)}
        self.crop_idx = {j: j - 1 for j in range(1, 42)}
        self.margin = self._margin_table()

        self.blocks, off = [], 0
        for p in self.sm.plots:
            for s in sorted(self.sm.season_support[p]):
                sup = list(self.sm.season_support[p][s])
                self.blocks.append((p, s, sup, off))
                off += len(sup)
        self.D = off

        self.mc_N = 2000
        self.mc_seed = 0
        self.mc_R = None
        self.mc_cache = {}

    def _margin_table(self):
        M = np.full((len(self.rev.units), 41), -np.inf)
        for iu in range(len(self.rev.units)):
            for jc in range(41):
                ts = self.rev.ts_map[iu, jc]
                if ts >= 0:
                    M[iu, jc] = self.rev.p[ts, jc] * self.rev.q[ts, jc] - self.rev.c[ts, jc]
        return M

    def _mg(self, plot, s, j):
        if j is None:
            return -np.inf
        u = self.sm.unit_of(plot, s)
        return float(self.margin[self.unit_idx[u], self.crop_idx[j]])

    def _eff_margin(self, plot, s, j, rem, mode=1):
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
        used = {j: 0.0 for j in range(1, 42)}
        for p, seasons in plan_y.items():
            for s, cs_ in seasons.items():
                for j, a in cs_:
                    u = self.sm.unit_of(p, s)
                    jc = self.crop_idx[j]
                    ts = self.rev.ts_map[self.unit_idx[u], jc]
                    used[j] += a * float(self.sm.unit_area[u]) * float(self.rev.q[ts, jc])
        return {j: float(self.rev.sales0[j - 1]) - used[j] for j in range(1, 42)}

    def _fill(self, p, s, cands, demand, mode=1, exclude=(), single=False, max_crops=4):
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
                    a = min(remain, rem / Y_u)
                    r = 1.0
                else:
                    a = remain
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
                break
            taken.add(j)
        return out, tot

    def _greedy_year(self, prev_sets, mode=1):
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
            else:
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

    def _adjacent(self, plan, p, y, s):
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
                        continue
                    for s, cs_ in plan[y][p].items():
                        sup = self.sm.season_support[p][s]
                        if not (set(sup) & BEAN):
                            continue
                        j = cs_[0][0] if cs_ else None
                        cand.append((self._mg(p, s, j) if j is not None else 0.0, y, s))
                if not cand:
                    continue
                _, y, s = min(cand)
                adj = self._adjacent(plan, p, y, s)
                ok = [j for j in self.sm.season_support[p][s]
                      if j in BEAN and j not in adj]
                if not ok:
                    continue
                nb = max(ok, key=lambda j: self._mg(p, s, j))

                if self.sm.plot_type[p] == 'D' and s == 1 and not plan[y][p].get(2):
                    adj2 = self._adjacent(plan, p, y, 2) | {nb}
                    ok2 = [j for j in S2 if j not in adj2]
                    if not ok2:
                        continue
                    plan[y][p][2] = [(max(ok2, key=lambda j: self._mg(p, 2, j)), 1.0)]
                plan[y][p][s] = [(nb, 1.0)]

    def _encode(self, plan_y):
        vec = np.zeros(self.D)
        for p, s, sup, off in self.blocks:
            for j, a in plan_y.get(p, {}).get(s, []):
                if j in sup:
                    vec[off + sup.index(j)] = a
        return vec

    def _decode(self, vec, y, prev_sets):
        plan = defaultdict(lambda: defaultdict(list))
        for p, s, sup, off in self.blocks:
            t = self.sm.plot_type[p]

            adj = set()
            if s == 1:
                ps2 = set(prev_sets.get(p, {}).get(2, []))
                ps1 = set(prev_sets.get(p, {}).get(1, []))
                adj = ps2 | (ps1 if not ps2 else set())
            else:
                adj = {j for j, _ in plan[p].get(1, [])}
            w = {j: max(0.0, vec[off + i]) for i, j in enumerate(sup)}
            if t == 'D' and s == 1:
                if RICE not in adj and w.get(RICE, 0.0) > 0.02:
                    plan[p][1].append((RICE, 1.0))
                    continue
                w.pop(RICE, None)
            if adj:
                for j in adj:
                    w.pop(j, None)
            total = sum(w.values())
            if total > 1.0:
                k = 1.0 / total
                w = {j: a * k for j, a in w.items()}
            for j, a in w.items():
                if a >= EPS:
                    plan[p][s].append((j, a))

        for p, seasons in list(plan.items()):
            if self.sm.plot_type[p] != 'D':
                continue
            s1 = {j for j, a in seasons.get(1, []) if a > 0}
            s2 = {j for j, a in seasons.get(2, []) if a > 0}
            if RICE in s1 and s1 - {RICE}:
                seasons[1] = [(RICE, next(a for j, a in seasons[1] if j == RICE))]
                s1 = {RICE}
            if RICE in s1 and s2:
                del seasons[2]; s2 = set()
            if s2 and not (s1 & WATER_VEG):
                del seasons[2]; s2 = set()
            if (s1 & WATER_VEG) and not s2:
                adj = set(s1)
                ok = [j for j in WATER_S2 if j not in adj]
                if ok:
                    seasons[2] = [(max(ok, key=lambda j: self._mg(p, 2, j)), 1.0)]

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
        for p in self.sm.plots:
            out.setdefault(p, {})
        return out

    def _n_crops(self, plan_y):
        return sum(1 for _, seas in plan_y.items()
                   for _, cs_ in seas.items() for _, a in cs_ if a > 0)

    @staticmethod
    def _bean_in(crop_sets_p):
        return any(s & BEAN for s in crop_sets_p.values())

    def _bean_close(self, plan_y, y, prev_sets, prev2_sets):
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
        n_enc = len(self.rc.check_year(plan_y, y))
        n_re = len(self.rc.replant_violations(prev_sets, self.rc.crop_sets(plan_y)))
        n_bean = self._bean_close(plan_y, y, prev_sets, prev2_sets)
        return LAM * (n_enc + n_re + n_bean) + SPARSE * self._n_crops(plan_y)

    def _fitness(self, plan_y, y, prev_sets, prev2_sets, problem, mode, dist, n_quad):
        x = self.sm.derive(plan_y)
        if problem == 1:
            profit = self.rev.profit_det(x, mode)
        elif problem == 3:

            profit = float(self.rev.profit_mc(x, y, mode, N=self.mc_N, R=self.mc_R,
                                              samples=self._mc_samples(y)).mean())
        else:
            profit = self.rev.profit_stoch(x, y, dist, mode, n_quad)
        return profit - self._virtual_penalty(plan_y, y, prev_sets, prev2_sets)

    def _mc_samples(self, y):
        if y not in self.mc_cache:
            self.mc_cache[y] = self.rev.mc_samples(y, N=self.mc_N, R=self.mc_R, seed=self.mc_seed)
        return self.mc_cache[y]

    def _need_bean_plots(self, y, prev_sets, prev2_sets):
        if y < 2025:
            return []
        return [p for p in self.sm.plots
                if not self._bean_in(prev2_sets.get(p, {}))
                and not self._bean_in(prev_sets.get(p, {}))]

    def _force_bean(self, vec, p, prev_sets):
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
                    vec[off + rice_i] = 0.0
            return True
        return False

    def _k2_warm(self, greedy_y, y, prev_sets, prev2_sets, problem, mode, dist, n_quad,
                 npop=32, ngen=60, F_=F, CR_=CR):
        slots = [(p, s, list(self.sm.season_support[p][s]))
                 for p in self.sm.plots for s in sorted(self.sm.season_support[p])]
        D2 = 4 * len(slots)
        rng = np.random.default_rng(self.seed * 1000 + y + 777)

        def enc2(plan_y):
            v = np.zeros(D2)
            for bi, (p, s, sup) in enumerate(slots):
                b = bi * 4
                for k, (j, a) in enumerate(plan_y.get(p, {}).get(s, [])[:2]):
                    if j in sup:
                        v[b + 2 * k] = sup.index(j)
                        v[b + 2 * k + 1] = min(a, 1.0)
            return v

        def dec2(v):
            plan = defaultdict(lambda: defaultdict(list))
            for bi, (p, s, sup) in enumerate(slots):
                b = bi * 4
                for k in range(2):
                    u = int(np.clip(round(v[b + 2 * k]), 0, len(sup) - 1))
                    a = float(np.clip(v[b + 2 * k + 1], 0.0, 1.0))
                    if a > EPS:
                        plan[p][s].append((sup[u], a))
            out = {p: dict(ss) for p, ss in plan.items()}
            for p in self.sm.plots:
                out.setdefault(p, {})
            return out

        pop = np.zeros((npop, D2))
        pop[0] = enc2(greedy_y)
        for i in range(1, npop):
            v = pop[0].copy()
            for bi, (p, s, sup) in enumerate(slots):
                b = bi * 4
                if v[b + 1] > 0:
                    v[b + 1] = float(np.clip(v[b + 1] + rng.uniform(-0.15, 0.15), 0.0, 1.0))
                if v[b + 3] > 0:
                    v[b + 3] = float(np.clip(v[b + 3] + rng.uniform(-0.15, 0.15), 0.0, 1.0))
            for _ in range(int(rng.integers(1, 5))):
                bi = int(rng.integers(len(slots)))
                b = bi * 4
                k = int(rng.integers(2))
                if v[b + 2 * k + 1] > 0.05:
                    v[b + 2 * k] = rng.integers(len(slots[bi][2]))
            pop[i] = v
        fits = [self._fitness(dec2(pop[i]), y, prev_sets, prev2_sets,
                              problem, mode, dist, n_quad) for i in range(npop)]
        for _ in range(ngen):
            for i in range(npop):
                a, b, c = rng.choice([j for j in range(npop) if j != i], 3, replace=False)
                trial = pop[i].copy()
                mask = rng.random(D2) < CR_
                mask[rng.integers(D2)] = True
                trial[mask] = (pop[a] + F_ * (pop[b] - pop[c]))[mask]
                for bi, (p, s, sup) in enumerate(slots):
                    b = bi * 4
                    for k in range(2):
                        trial[b + 2 * k] = float(np.clip(trial[b + 2 * k], 0, len(sup) - 1))
                        trial[b + 2 * k + 1] = float(np.clip(trial[b + 2 * k + 1], 0.0, 1.0))
                ft = self._fitness(dec2(trial), y, prev_sets, prev2_sets,
                                   problem, mode, dist, n_quad)
                if ft > fits[i]:
                    pop[i], fits[i] = trial, ft
        return dec2(pop[int(np.argmax(fits))])

    def _de_year(self, vec0, y, prev_sets, prev2_sets, problem, mode, dist, n_quad,
                 npop=NP, ngen=G, F_=F, CR_=CR):
        rng = np.random.default_rng(self.seed * 1000 + y)
        D = self.D
        need_bean = self._need_bean_plots(y, prev_sets, prev2_sets)
        pop = np.zeros((npop, D))
        pop[0] = vec0
        for i in range(1, npop):
            v = vec0.copy()
            nz = np.nonzero(vec0)[0]
            if len(nz):
                v[nz] = np.clip(v[nz] + rng.uniform(-0.15, 0.15, len(nz)), 0.0, 1.0)
            for _ in range(int(rng.integers(1, 4))):
                b = int(rng.integers(len(self.blocks)))
                _, _, sup, off = self.blocks[b]
                v[off + int(rng.integers(len(sup)))] = rng.uniform(0.2, 1.0)
            pop[i] = np.clip(v, 0.0, 1.0)
        for i in range(npop):
            for p in need_bean:
                self._force_bean(pop[i], p, prev_sets)
        fits = [self._fitness(self._decode(pop[i], y, prev_sets),
                              y, prev_sets, prev2_sets, problem, mode, dist, n_quad)
                for i in range(npop)]
        for _ in range(ngen):
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

    def _candidates(self, p, prev_sets, demand, mode=1):
        t = self.sm.plot_type[p]

        def top(s, cands, n=3, exclude=()):
            return sorted([j for j in cands if j not in exclude],
                          key=lambda j: -self._eff_margin(p, s, j, demand[j], mode))[:n]

        outs = [{}]
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
        else:
            ex = prev_sets.get(2, set())
            for s1 in top(1, VEG, 3, exclude=ex):
                for s2 in top(2, VEG, 3, exclude={s1}):
                    outs.append({1: [(s1, 1.0)], 2: [(s2, 1.0)]})
        return outs

    def _polish(self, plan, problem, mode, dist, n_quad, max_iter=6):
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

    def _profit(self, plan, y, problem, mode, dist, n_quad):
        x = self.sm.derive(plan[y])
        if problem == 1:
            return self.rev.profit_det(x, mode)
        if problem == 3:
            return float(self.rev.profit_mc(x, y, mode, N=self.mc_N, R=self.mc_R,
                                            samples=self._mc_samples(y)).mean())
        return self.rev.profit_stoch(x, y, dist=dist, mode=mode, n_quad=n_quad)

    def fitness(self, plan, problem=1, mode=1, dist='normal', n_quad=40, base=None):
        self.rc.raise_if_invalid(plan, base=base)
        return sum(self._profit(plan, y, problem, mode, dist, n_quad) for y in YEARS)

    def report(self, plan, mode=1, problem=1):
        for y in YEARS:
            x = self.sm.derive(plan[y])
            area = sum(self.sm.unit_area[u] for u in {u for u, _ in x})
            if problem == 3:
                pi = self.rev.profit_mc(x, y, mode, N=self.mc_N, R=self.mc_R,
                                        samples=self._mc_samples(y))
                print(f'{y}: MC 期望 {pi.mean()/1e4:.2f} 万  标准差 {pi.std()/1e4:.2f} 万'
                      f'  面积 {area:.1f} 亩')
            else:
                det = self.rev.profit_det(x, mode) / 1e4
                stoch = self.rev.profit_stoch(x, y, dist='normal', mode=mode) / 1e4
                print(f'{y}: 确定性 {det:.2f} 万  期望 {stoch:.2f} 万  面积 {area:.1f} 亩')
        n1 = sum(len(self.rc.replant_violations(self.rc.crop_sets(plan[t - 1]),
                                                self.rc.crop_sets(plan[t])))
                 for t in range(2024, 2031))
        n2 = len(self.rc.bean_violations(plan))
        print(f'重茬违规={n1}, 豆类窗口违规={n2}')

    def solve(self, baseline=None, problem=1, mode=1, dist='normal',
              n_quad=40, seed=None, npop=NP, ngen=G, verbose=True, k2_warm=False,
              mc_N=2000, mc_seed=0):
        if seed is not None:
            self.seed = seed
            self.rng = random.Random(self.seed)
        self.mc_N = mc_N
        self.mc_seed = mc_seed
        self.mc_R = self.rev.corr_matrix() if problem == 3 else None
        self.mc_cache = {}
        plan = {2023: baseline or {}}
        base_cs = self.rc.crop_sets(plan[2023])
        greedy = {}
        prev = base_cs
        for y in YEARS:
            greedy[y] = self._greedy_year(prev, mode)
            prev = self.rc.crop_sets(greedy[y])
        prev = base_cs
        for y in YEARS:
            p2 = base_cs if y - 2 == 2023 else (self.rc.crop_sets(plan[y - 2])
                                                if y - 2 >= 2024 else {})
            if k2_warm:

                k2_plan = self._k2_warm(greedy[y], y, prev, p2, problem, mode, dist, n_quad)
                vec0 = self._encode(k2_plan)
            else:
                vec0 = self._encode(greedy[y])
            best, f = self._de_year(vec0, y, prev, p2, problem, mode, dist, n_quad,
                                    npop=npop, ngen=ngen)
            plan[y] = self._decode(best, y, prev)
            prev = self.rc.crop_sets(plan[y])
            if mode == 2:
                self._repair_beans(plan)
                prev = self.rc.crop_sets(plan[y])
            if verbose:
                print(f'  {y}: DE 最优适应度 = {f/1e4:.2f} 万元')
        self._repair_beans(plan)
        self._polish(plan, problem, mode, dist, n_quad, max_iter=6)
        self._repair_beans(plan)

        full = integrate({y: plan[y] for y in YEARS}, base=plan[2023])
        self.rc.raise_if_invalid(full, base=plan[2023])
        fit = sum(self._profit(plan, y, problem, mode, dist, n_quad) for y in YEARS)
        return plan, fit

def load_2023_baseline(scheme=None):
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
    warm = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False
    problem = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    n_quad = int(sys.argv[6]) if len(sys.argv) > 6 else 16
    mc_N = int(sys.argv[7]) if len(sys.argv) > 7 else 2000
    mc_seed = int(sys.argv[8]) if len(sys.argv) > 8 else 0
    base = load_2023_baseline()

    print(f'全量分配编码变量数 D = {FullDE().D}   K=2 热启动 = {warm}   问题 = {problem}'
          f'   n_quad={n_quad}   mc_N={mc_N}')
    de = FullDE(seed=0)
    plan, fit = de.solve(baseline=base, problem=problem, mode=mode, npop=npop, ngen=ngen,
                         k2_warm=warm, n_quad=n_quad, mc_N=mc_N, mc_seed=mc_seed)
    print(f'\nFullDE 问题{problem} mode{mode} 总适应度 = {fit/1e4:.2f} 万元')
    de.report(plan, mode=mode, problem=problem)

    de.rc.raise_if_invalid(integrate({y: plan[y] for y in YEARS}, base=base), base=base)
    print(f'规则检查（{npop}×{ngen}，K2预热={warm}）：全部通过 ✓，方案已保存')
    tag = '_k2warm' if warm else ''
    fname = f'de_plan_full_p{problem}_mode{mode}{tag}.pkl'
    pickle.dump(plan, open(DATA / fname, 'wb'))
