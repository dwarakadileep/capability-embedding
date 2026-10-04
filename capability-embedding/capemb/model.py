"""Formal model: atoms (conditions), states, goals, capabilities and symbolic composition.

This module is the *symbolic ground truth*. The embedding in embedding.py is designed so that
its vector operations agree with the symbolic operations defined here.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

TYPES = ["API", "DATABASE", "GUI", "EVENT", "FUNCTION", "FILE", "COMPUTATION", "MESSAGE", "SERVICE"]
COST_KEYS = ["time_ms", "money", "resource", "risk", "energy"]

_CMP = re.compile(r"^\s*([A-Za-z_][\w.]*)\s*(==|!=|>=|<=|>|<)\s*(.+?)\s*$")
_IN = re.compile(r"^\s*([A-Za-z_][\w.]*)\s+in\s+(\{.*\})\s*$")


def _parse_val(s: str):
    s = s.strip()
    if s.startswith("{"):
        return frozenset(_parse_val(x) for x in s.strip("{}").split(",") if x.strip())
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s.strip("\"'")


@dataclass(frozen=True)
class Atom:
    """A single atomic condition  var <op> value  (op in ==, !=, >, >=, <, <=, in)."""
    var: str
    op: str
    val: Any

    @staticmethod
    def parse(s: str) -> "Atom":
        m = _IN.match(s)
        if m:
            return Atom(m.group(1), "in", _parse_val(m.group(2)))
        m = _CMP.match(s)
        if not m:
            raise ValueError(f"cannot parse condition: {s!r}")
        return Atom(m.group(1), m.group(2), _parse_val(m.group(3)))

    @property
    def key(self) -> str:
        v = "{" + ",".join(sorted(map(str, self.val))) + "}" if self.op == "in" else str(self.val)
        return f"{self.var}{self.op}{v}"

    def holds_value(self, x) -> bool:
        if x is None:
            return False
        try:
            if self.op == "==":
                return x == self.val
            if self.op == "!=":
                return x != self.val
            if self.op == "in":
                return x in self.val
            if isinstance(x, bool):
                return False
            return {">": x > self.val, ">=": x >= self.val, "<": x < self.val, "<=": x <= self.val}[self.op]
        except TypeError:
            return False

    def entails(self, other: "Atom") -> bool:
        """Does asserting self guarantee other?  (x==5 entails x>0, x==A entails x in {A,B}, ...)"""
        if self == other:
            return True
        return self.op == "==" and self.var == other.var and other.holds_value(self.val)

    def witness(self):
        """A value of var that satisfies this atom."""
        v = self.val
        if self.op in ("==", ">=", "<="):
            return v
        if self.op == ">":
            return v + 1
        if self.op == "<":
            return v - 1
        if self.op == "!=":
            return (not v) if isinstance(v, bool) else f"not_{v}"
        return sorted(v, key=str)[0]


def _atoms(xs):
    return [x if isinstance(x, Atom) else Atom.parse(x) for x in xs]


@dataclass
class State:
    vars: dict
    data: set = field(default_factory=set)  # available data items, keys "name:type"

    def satisfies(self, atoms) -> bool:
        return all(a.holds_value(self.vars.get(a.var)) for a in atoms)

    def copy(self):
        return State(dict(self.vars), set(self.data))


@dataclass
class Goal:
    atoms: list

    def __post_init__(self):
        self.atoms = _atoms(self.atoms)


@dataclass
class Capability:
    name: str
    type: str = "FUNCTION"
    inputs: list = field(default_factory=list)       # (name, type, domain, required)
    outputs: list = field(default_factory=list)      # (name, type, domain)
    pre: list = field(default_factory=list)
    eff: list = field(default_factory=list)
    constraints: list = field(default_factory=list)
    resources: list = field(default_factory=list)
    cost: dict = field(default_factory=dict)         # time_ms, money, resource, risk, energy
    reliability: float = 1.0
    availability: float = 1.0
    window: tuple | None = None                      # availability window (hour_start, hour_end)
    mechanism: dict = field(default_factory=dict)
    components: tuple = ()
    types: tuple = ()
    mechs: tuple = ()
    valid: bool = True
    notes: list = field(default_factory=list)

    def __post_init__(self):
        self.pre, self.eff, self.constraints = _atoms(self.pre), _atoms(self.eff), _atoms(self.constraints)
        self.inputs = [tuple(i) if len(i) == 4 else (*i, True) for i in self.inputs]
        self.outputs = [tuple(o) for o in self.outputs]
        self.cost = {k: float(self.cost.get(k, 0.0)) for k in COST_KEYS}
        if not self.types:
            self.types = (self.type,)
        if not self.mechs:
            self.mechs = (dict(self.mechanism),)

    # ---- data interface -------------------------------------------------------------
    @property
    def in_keys(self) -> dict:
        """key -> weight (required = 1.0, optional = 0.5)"""
        out = {}
        for n, t, _d, req in self.inputs:
            out[f"{n}:{t}"] = max(out.get(f"{n}:{t}", 0.0), 1.0 if req else 0.5)
        return out

    @property
    def out_keys(self) -> set:
        return {f"{n}:{t}" for n, t, *_ in self.outputs}

    def parts(self) -> tuple:
        return self.components or (self,)

    def availability_at(self, t: float | None = None) -> float:
        if self.components:
            return min(c.availability_at(t) for c in self.components)
        if self.window is not None and t is not None:
            lo, hi = self.window
            return float(lo <= (t % 24) < hi) * self.availability
        return float(self.availability)


# ---------------------------------------------------------------------------------------
# symbolic relations (ground truth used to validate the vector design)
# ---------------------------------------------------------------------------------------
def symbolic_link(a: Capability, b: Capability) -> bool:
    """a's effects/outputs satisfy at least one precondition/input of b."""
    return any(e.entails(p) for e in a.eff for p in b.pre) or bool(a.out_keys & set(b.in_keys))


def symbolic_conflict(a: Capability, b: Capability) -> bool:
    """a's effects contradict a precondition of b."""
    return any(e.var == p.var and e.op == "==" and not p.holds_value(e.val) for e in a.eff for p in b.pre)


def symbolic_composable(a: Capability, b: Capability) -> bool:
    return symbolic_link(a, b) and not symbolic_conflict(a, b)


def compose_pair(a: Capability, b: Capability) -> Capability:
    """Formal b o a  (a runs first)."""
    pre = list(a.pre)
    for p in b.pre:
        if not any(e.entails(p) for e in a.eff) and p not in pre:
            pre.append(p)
    vars_b = {e.var for e in b.eff}
    eff = list(b.eff) + [e for e in a.eff if e.var not in vars_b and e not in b.eff]
    cons = list(a.constraints) + [k for k in b.constraints if k not in a.constraints]
    ins = dict(a.in_keys)
    for k, w in b.in_keys.items():
        if k not in a.out_keys:
            ins[k] = max(ins.get(k, 0.0), w)
    inputs = [(k.split(":")[0], k.split(":")[1], "", w >= 1.0) for k, w in ins.items()]
    outputs = [(k.split(":")[0], k.split(":")[1], "") for k in sorted(a.out_keys | b.out_keys)]
    cost = {k: a.cost[k] + b.cost[k] for k in COST_KEYS}
    cost["risk"] = 1 - (1 - a.cost["risk"]) * (1 - b.cost["risk"])
    return Capability(
        name=f"{a.name}>{b.name}", type="COMPOSITE", inputs=inputs, outputs=outputs, pre=pre, eff=eff,
        constraints=cons, resources=sorted(set(a.resources) | set(b.resources)), cost=cost,
        reliability=a.reliability * b.reliability, components=a.parts() + b.parts(),
        types=a.types + b.types, mechs=a.mechs + b.mechs,
    )


class CompositionError(ValueError):
    pass


def compose_formal(caps: list[Capability], strict: bool = True) -> Capability:
    """Formal C_n o ... o C_1 for caps = [C_1, ..., C_n]; validity is checked step by step
    against the composite built so far (so C3 may be fed by C1 or C2)."""
    acc, notes, ok = caps[0], [], True
    for nxt in caps[1:]:
        if not symbolic_link(acc, nxt):
            notes.append(f"no link {acc.name} -> {nxt.name}")
            ok = False
        if symbolic_conflict(acc, nxt):
            notes.append(f"conflict {acc.name} -> {nxt.name}")
            ok = False
        if strict and not ok:
            raise CompositionError("; ".join(notes))
        acc = compose_pair(acc, nxt)
    if len(caps) == 1:
        acc = compose_pair(acc, Capability(name="id"))  # identity-like wrapper keeps API uniform
        acc.name = caps[0].name
    acc.valid, acc.notes = ok, notes
    return acc


# ---------------------------------------------------------------------------------------
# simulator (used to semantically verify composites)
# ---------------------------------------------------------------------------------------
def applicable(state: State, c: Capability) -> bool:
    return state.satisfies(c.pre + c.constraints) and set(c.in_keys) <= state.data | {k for k, w in c.in_keys.items() if w < 1.0}


def apply(state: State, c: Capability) -> State:
    s = state.copy()
    for e in c.eff:
        s.vars[e.var] = e.val
    s.data |= c.out_keys
    return s


def witness_state(c: Capability) -> State:
    s = State({}, {k for k, w in c.in_keys.items() if w >= 1.0})
    for a in c.pre + c.constraints:
        s.vars[a.var] = a.witness()
    return s


def run_chain(state: State, caps: list[Capability]):
    for c in caps:
        if not applicable(state, c):
            return None
        state = apply(state, c)
    return state
