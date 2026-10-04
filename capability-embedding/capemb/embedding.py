"""Block-structured capability embedding (see REPORT.md for the full derivation).

phi(C) = [ P | E | K | I | O | T | M | R | Q | N ]   (raw, un-weighted blocks)

  P,E,K  in R^A   preconditions / effects / constraints over a vocabulary of A atomic conditions
  I,O    in R^D   required(1.0)/optional(0.5) inputs and outputs over D data items
  T      in R^9   capability type (mean one-hot over components)
  M      in R^16  feature-hashed execution mechanism (mean over components)
  R      in R^Rn  multi-hot resource requirements
  Q      in R^7   [time_ms, money, resource, risk, energy, -ln(reliability), unavailable]
  N      in R^1   number of atomic components

States are phi_S(S) = [holds(atom_1..A) | has(data_1..D)] and goals phi_G(G) = indicator over atoms.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .model import COST_KEYS, TYPES, Atom, Capability, Goal, State, compose_formal

M_DIM = 16
Q_NAMES = ["time_ms", "money", "resource", "risk", "energy", "neglog_rel", "unavail"]

VIEWS = {  # block weights used by cosine similarity
    "function":  {"E": 1.0, "O": 1.0},                                   # what it does
    "interface": {"P": 1.0, "E": 1.0, "K": 1.0, "I": 1.0, "O": 1.0},    # what it needs + does
    "full":      {"P": 1.0, "E": 1.0, "K": 1.0, "I": 1.0, "O": 1.0, "T": 0.5, "M": 0.5, "R": 0.25},
}
COST_W = np.array([1e-3, 10.0, 0.5, 2.0, 1.0])  # weights collapsing cost attributes to one scalar


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


class Vocabulary:
    """Fixed vocabularies + the three structural matrices used for entailment/conflict/override."""

    def __init__(self, caps: list[Capability], goals: list[Goal] = ()):
        atoms: dict[str, Atom] = {}
        data, res = [], []
        for c in caps:
            for a in c.pre + c.eff + c.constraints:
                atoms.setdefault(a.key, a)
            for k in list(c.in_keys) + sorted(c.out_keys):
                if k not in data:
                    data.append(k)
            for r in c.resources:
                if r not in res:
                    res.append(r)
        for g in goals:
            for a in g.atoms:
                atoms.setdefault(a.key, a)
        self.atoms = list(atoms.values())
        self.aidx = {a.key: i for i, a in enumerate(self.atoms)}
        self.data, self.didx = data, {k: i for i, k in enumerate(data)}
        self.res, self.ridx = res, {k: i for i, k in enumerate(res)}
        A = len(self.atoms)
        vars_ = sorted({a.var for a in self.atoms})
        vidx = {v: i for i, v in enumerate(vars_)}
        self.Ent = np.zeros((A, A))   # Ent[e,p]=1  <=>  asserting atom e guarantees atom p
        self.X = np.zeros((A, A))     # X[e,p]=1    <=>  asserting e contradicts p (same variable)
        self.Vm = np.zeros((A, len(vars_)))  # atom -> variable membership
        for i, e in enumerate(self.atoms):
            self.Vm[i, vidx[e.var]] = 1
            for j, p in enumerate(self.atoms):
                if e.entails(p):
                    self.Ent[i, j] = 1
                elif e.var == p.var and e.op == "==" and not p.holds_value(e.val):
                    self.X[i, j] = 1


@dataclass
class Encoded:
    kind: str        # 'state' | 'goal' | 'capability'
    vec: np.ndarray
    obj: Any = None
    name: str = ""


class Embedder:
    def __init__(self, vocab: Vocabulary):
        self.v = vocab
        self.A, self.D, self.Rn = len(vocab.atoms), len(vocab.data), len(vocab.res)
        sizes = [("P", self.A), ("E", self.A), ("K", self.A), ("I", self.D), ("O", self.D),
                 ("T", len(TYPES)), ("M", M_DIM), ("R", self.Rn), ("Q", len(Q_NAMES)), ("N", 1)]
        self.sl, off = {}, 0
        for n, s in sizes:
            self.sl[n] = slice(off, off + s)
            off += s
        self.dim = off

    # ------------------------------------------------------------------ helpers
    def blk(self, x: Encoded | np.ndarray, name: str):
        v = x.vec if isinstance(x, Encoded) else x
        return v[..., self.sl[name]]

    @staticmethod
    def _mech_vec(m: dict) -> np.ndarray:
        v = np.zeros(M_DIM)
        for k, val in m.items():
            h = int(hashlib.md5(f"{k}={val}".encode()).hexdigest(), 16)
            v[h % M_DIM] += 1.0 if (h >> 64) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def _closure(self, E):  # effects plus everything they entail
        return np.clip(E @ self.v.Ent, 0, 1)

    # ------------------------------------------------------------------ encode(...)
    def encode_state(self, s: State) -> Encoded:
        a = np.array([1.0 if at.holds_value(s.vars.get(at.var)) else 0.0 for at in self.v.atoms])
        d = np.array([1.0 if k in s.data else 0.0 for k in self.v.data])
        return Encoded("state", np.concatenate([a, d]), s)

    def encode_goal(self, g: Goal) -> Encoded:
        v = np.zeros(self.A)
        for a in g.atoms:
            v[self.v.aidx[a.key]] = 1.0
        return Encoded("goal", v, g)

    def encode_capability(self, c: Capability, t: float | None = None) -> Encoded:
        v, sl, vc = np.zeros(self.dim), self.sl, self.v
        for blk, atoms in (("P", c.pre), ("E", c.eff), ("K", c.constraints)):
            for a in atoms:
                v[sl[blk]][vc.aidx[a.key]] = 1.0
        for k, w in c.in_keys.items():
            v[sl["I"]][vc.didx[k]] = w
        for k in c.out_keys:
            v[sl["O"]][vc.didx[k]] = 1.0
        for ty in c.types:
            v[sl["T"]][TYPES.index(ty)] += 1.0 / len(c.types)
        for m in c.mechs:
            v[sl["M"]] += self._mech_vec(m) / len(c.mechs)
        for r in c.resources:
            v[sl["R"]][vc.ridx[r]] = 1.0
        q = [c.cost[k] for k in COST_KEYS] + [-math.log(max(c.reliability, 1e-12)), 1.0 - c.availability_at(t)]
        v[sl["Q"]] = q
        v[sl["N"]] = len(c.parts())
        return Encoded("capability", v, c, c.name)

    def encode(self, x, **kw) -> Encoded:
        if isinstance(x, State):
            return self.encode_state(x)
        if isinstance(x, Goal):
            return self.encode_goal(x)
        if isinstance(x, Capability):
            return self.encode_capability(x, **kw)
        raise TypeError(type(x))

    # ------------------------------------------------------------------ compose(...)
    def compose(self, caps: list[Capability | Encoded], strict: bool = True, t=None) -> Encoded:
        """Composite capability C_n o ... o C_1 (list is in execution order).
        The encoding is produced with the *vector algebra* compose_vectors and the formal
        composite is attached as .obj so the two can be cross-checked."""
        objs = [c.obj if isinstance(c, Encoded) else c for c in caps]
        formal = compose_formal(objs, strict=strict)
        vecs = [(c if isinstance(c, Encoded) else self.encode_capability(c, t)).vec for c in caps]
        out = self.compose_vectors(vecs)
        return Encoded("capability", out, formal, formal.name)

    def compose_vectors(self, vecs: list[np.ndarray]) -> np.ndarray:
        acc = vecs[0]
        for b in vecs[1:]:
            acc = self._compose2(acc, b)
        return acc

    def _compose2(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        s, vc = self.sl, self.v
        g = lambda x, n: x[s[n]]
        out = np.zeros_like(a)
        Ecl1 = self._closure(g(a, "E"))
        out[s["P"]] = np.maximum(g(a, "P"), g(b, "P") * (1 - Ecl1))                # unmet preconditions bubble up
        over = np.clip((g(b, "E") @ vc.Vm) @ vc.Vm.T, 0, 1)                         # variables rewritten by b
        out[s["E"]] = np.maximum(g(b, "E"), g(a, "E") * (1 - over))                 # later effect wins
        out[s["K"]] = np.maximum(g(a, "K"), g(b, "K"))
        out[s["I"]] = np.maximum(g(a, "I"), g(b, "I") * (1 - g(a, "O")))            # inputs already produced vanish
        out[s["O"]] = np.maximum(g(a, "O"), g(b, "O"))
        na, nb = g(a, "N")[0], g(b, "N")[0]
        for n in ("T", "M"):
            out[s[n]] = (na * g(a, n) + nb * g(b, n)) / (na + nb)                   # size-weighted mean
        out[s["R"]] = np.maximum(g(a, "R"), g(b, "R"))
        qa, qb, q = g(a, "Q"), g(b, "Q"), np.zeros(len(Q_NAMES))
        q[[0, 1, 2, 4, 5]] = qa[[0, 1, 2, 4, 5]] + qb[[0, 1, 2, 4, 5]]               # additive (log-reliability adds)
        q[3] = 1 - (1 - qa[3]) * (1 - qb[3])                                         # independent risks
        q[6] = max(qa[6], qb[6])                                                     # unavailable if any part is
        out[s["Q"]] = q
        out[s["N"]] = na + nb
        return out

    # ------------------------------------------------------------------ compatibility (pairwise, vectorised)
    def compat_matrix(self, xs: list[Encoded], ys: list[Encoded] | None = None) -> dict:
        """M[i,j]: how much of y_j's requirements are supplied by x_i, and whether x_i contradicts y_j.

        feed     = (closure(E_i) ++ O_i) . (P_j ++ I_j) / |P_j ++ I_j|       in [0,1]
        conflict = (clip(E_i X)) . P_j / |P_j|                                in [0,1]
        score    = feed * (1 - conflict);   composable  <=>  feed>0 and conflict==0
        """
        ys = xs if ys is None else ys
        Va, Vb, s = np.stack([x.vec for x in xs]), np.stack([y.vec for y in ys]), self.sl
        E, O = Va[:, s["E"]], Va[:, s["O"]]
        prov = np.hstack([self._closure(E), O])
        req = np.hstack([Vb[:, s["P"]], Vb[:, s["I"]]])
        rs = req.sum(1)
        feed = prov @ req.T / np.where(rs > 0, rs, 1.0)[None, :]
        feed[:, rs == 0] = 0.0
        Pb = Vb[:, s["P"]]
        ps = Pb.sum(1)
        conflict = np.clip(E @ self.v.X, 0, 1) @ Pb.T / np.where(ps > 0, ps, 1.0)[None, :]
        Ib = Vb[:, s["I"]]
        ins = Ib.sum(1)
        feed_pre = self._closure(E) @ Pb.T / np.where(ps > 0, ps, 1.0)[None, :]   # state-condition channel only
        feed_io = O @ Ib.T / np.where(ins > 0, ins, 1.0)[None, :]                   # data channel only
        return dict(feed=feed, feed_pre=feed_pre, feed_io=feed_io, conflict=conflict,
                    score=feed * (1 - conflict), composable=(feed > 1e-9) & (conflict < 1e-9))

    def compat(self, a: Encoded, b: Encoded) -> dict:
        m = self.compat_matrix([a], [b])
        return {k: (bool(v[0, 0]) if k == "composable" else float(v[0, 0])) for k, v in m.items()}

    # ------------------------------------------------------------------ state / goal / capability relations
    def applicability(self, st: Encoded, c: Encoded) -> dict:
        sa, sd = st.vec[: self.A], st.vec[self.A:]
        P, K, I = self.blk(c, "P"), self.blk(c, "K"), self.blk(c, "I")
        total = P.sum() + K.sum() + I.sum()
        got = P @ sa + K @ sa + I @ sd
        frac = float(got / total) if total > 0 else 1.0
        return dict(score=frac, applicable=frac > 1 - 1e-9)

    def goal_progress(self, st: Encoded, g: Encoded) -> float:
        return float(g.vec @ st.vec[: self.A] / max(g.vec.sum(), 1))

    def _prov(self, c: Encoded):
        return np.concatenate([self._closure(self.blk(c, "E")), self.blk(c, "O")])

    def _req(self, c: Encoded):
        return np.concatenate([self.blk(c, "P"), self.blk(c, "I")])

    def relevance_all(self, goal: Encoded, st: Encoded, library: list[Encoded], depth=3, gamma=0.5) -> np.ndarray:
        """Backward 'need propagation' in vector space: layer 0 = unmet goal atoms; a capability
        that provides part of layer k contributes its *unmet* requirements to layer k+1.
        relevance = max_k gamma^k * (provided share of layer k)."""
        need = np.concatenate([goal.vec * (1 - st.vec[: self.A]), np.zeros(self.D)])
        provs, reqs = [self._prov(c) for c in library], [self._req(c) for c in library]
        score = np.zeros(len(library))
        for k in range(depth):
            if need.sum() <= 0:
                break
            contrib = np.array([p @ need / need.sum() for p in provs])
            score = np.maximum(score, gamma ** k * contrib)
            nxt = np.zeros_like(need)
            for r, cv in zip(reqs, contrib):
                if cv > 0:
                    nxt = np.maximum(nxt, r * (1 - st.vec))
            need = nxt
        return score

    def cost_scalar(self, c: Encoded) -> float:
        return float(self.blk(c, "Q")[:5] @ COST_W)

    def utility(self, c: Encoded, relevance: float) -> float:
        q = self.blk(c, "Q")
        return float(relevance * math.exp(-q[5]) * (1 - q[6]) / (1 + self.cost_scalar(c)))

    # ------------------------------------------------------------------ similarity(x, y)
    def _qnorm(self, c: Encoded):
        q = self.blk(c, "Q")
        return np.array([math.log1p(q[0]) / math.log1p(1e4), min(q[1], 1), min(q[2] / 10, 1), q[3],
                         min(q[4] / 10, 1), min(q[5], 1), q[6]])

    def similarity(self, x: Encoded, y: Encoded, view: str = "full", **kw) -> float:
        kinds = {x.kind, y.kind}
        if x.kind == y.kind == "capability":
            if view == "operational":
                return float(max(0.0, 1 - np.linalg.norm(self._qnorm(x) - self._qnorm(y)) / math.sqrt(len(Q_NAMES))))
            w = VIEWS[view]
            f = lambda c: np.concatenate([w.get(n, 0.0) * self.blk(c, n) for n in self.sl])
            return _cos(f(x), f(y))
        if x.kind == y.kind:
            return _cos(x.vec, y.vec)
        if kinds == {"state", "goal"}:
            st, g = (x, y) if x.kind == "state" else (y, x)
            return self.goal_progress(st, g)
        if kinds == {"state", "capability"}:
            st, c = (x, y) if x.kind == "state" else (y, x)
            return self.applicability(st, c)["score"]
        g, c = (x, y) if x.kind == "goal" else (y, x)  # goal vs capability: direct contribution
        return float(self._prov(c)[: self.A] @ g.vec / max(g.vec.sum(), 1))
