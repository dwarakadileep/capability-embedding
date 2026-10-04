"""Runs all required experiments and the evaluation. Usage:  python experiments.py"""
import itertools
import json
import os
import random
import time

import numpy as np

from capemb import Embedder, Vocabulary, load, symbolic_composable, compose_formal
from capemb.datasets import dump
from capemb.model import apply, applicable, run_chain, witness_state

OUT = {}
os.makedirs("results", exist_ok=True)


def setup(name):
    p = load(name)
    emb = Embedder(Vocabulary(list(p.caps.values()), [p.goal]))
    enc = {n: emb.encode(c) for n, c in p.caps.items()}
    return p, emb, enc


def auc(scores, labels):
    """ROC-AUC via the Mann-Whitney U statistic."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def best_acc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    return float(max(((s > t) == y).mean() for t in np.unique(s)))


def header(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


dump()

# ------------------------------------------------------------------ EXP 1: compatibility
header("EXP 1  Capability compatibility (assignment pattern: C1=CreateOrder, C2=MakePayment, C3=CancelCart)")
p, emb, enc = setup("assignment_pattern")
c1, c2, c3 = enc["CreateOrder"], enc["MakePayment"], enc["CancelCart"]
r12, r13 = emb.compat(c1, c2), emb.compat(c1, c3)
print(f"C1->C2  {r12}\nC1->C3  {r13}")
print(f"cosine(full) C1,C2 = {emb.similarity(c1, c2):.3f}   C1,C3 = {emb.similarity(c1, c3):.3f}   (similarity alone cannot decide)")
OUT["exp1"] = dict(c1_c2=r12, c1_c3=r13,
                   cos_c1_c2=emb.similarity(c1, c2), cos_c1_c3=emb.similarity(c1, c3),
                   passed=bool(r12["composable"] and not r13["composable"] and r13["conflict"] > 0))
print("PASS" if OUT["exp1"]["passed"] else "FAIL")

# ------------------------------------------------------------------ EXP 2: composition
header("EXP 2  Capability composition  C1 -> C2 -> C3  (ecommerce)")
p, emb, enc = setup("ecommerce")
chain = ["CreateOrder_API", "MakePayment_Gateway", "SendNotification"]
comp = emb.compose([enc[n] for n in chain])
direct = emb.encode(comp.obj)
diff = float(np.abs(comp.vec - direct.vec).max())
f = comp.obj
print(f"composite pre : {[a.key for a in f.pre]}")
print(f"composite eff : {[a.key for a in f.eff]}")
print(f"composite in  : {dict(f.in_keys)}  out: {sorted(f.out_keys)}")
print(f"valid={f.valid}  reliability={f.reliability:.4f}  time={f.cost['time_ms']:.0f}ms  risk={f.cost['risk']:.3f}")
print(f"max |vector-compose - encode(formal composite)| = {diff:.2e}")
sims = {n: dict(function=emb.similarity(comp, enc[n], "function"), full=emb.similarity(comp, enc[n], "full")) for n in chain}
for n, s in sims.items():
    print(f"  cos(composite, {n:22s}) function={s['function']:.3f} full={s['full']:.3f}")
rev = None
try:
    emb.compose([enc[n] for n in reversed(chain)])
except Exception as e:  # noqa
    rev = str(e)
print("reversed order rejected:", rev)
left = emb.compose_vectors([emb.compose_vectors([enc[chain[0]].vec, enc[chain[1]].vec]), enc[chain[2]].vec])
right = emb.compose_vectors([enc[chain[0]].vec, emb.compose_vectors([enc[chain[1]].vec, enc[chain[2]].vec])])
assoc = float(np.abs(left - right).max())
print(f"associativity gap  (C3.C2).C1 vs C3.(C2.C1) = {assoc:.2e}")
warm, goal = emb.encode(p.states["warm"]), emb.encode(p.goal)
print(f"applicable in warm state: {emb.applicability(warm, comp)}   goal progress before: {emb.goal_progress(warm, goal):.2f}")
after = apply(apply(apply(p.states['warm'], p.caps[chain[0]]), p.caps[chain[1]]), p.caps[chain[2]])
print(f"goal progress after executing chain: {emb.goal_progress(emb.encode(after), goal):.2f}")
# composites are first-class: can be fed to compat/similarity like atomic capabilities
print("composite is itself composable into GenerateReport?", emb.compat(comp, enc["GenerateReport"])["composable"])
OUT["exp2"] = dict(max_diff=diff, associativity_gap=assoc, similarity=sims, reversed_rejected=rev is not None,
                   reliability=f.reliability, goal_progress_after=emb.goal_progress(emb.encode(after), goal))

# ------------------------------------------------------------------ EXP 3: alternative implementations
header("EXP 3  Alternative implementations (CreateOrder via API / DB / GUI)")
alts = ["CreateOrder_API", "CreateOrder_DB", "CreateOrder_GUI"]
rows = []
for a, b in itertools.combinations(alts, 2):
    row = dict(pair=f"{a}~{b}", function=emb.similarity(enc[a], enc[b], "function"), full=emb.similarity(enc[a], enc[b], "full"),
               identical_vec=bool(np.allclose(enc[a].vec, enc[b].vec)), composable=emb.compat(enc[a], enc[b])["composable"])
    rows.append(row)
    print(f"{row['pair']:40s} function={row['function']:.3f} full={row['full']:.3f} identical={row['identical_vec']} composable_with_each_other={row['composable']}")
ref = emb.similarity(enc["CreateOrder_API"], enc["MakePayment_Gateway"], "full")
print(f"reference: full cos(CreateOrder_API, MakePayment_Gateway) = {ref:.3f}")
OUT["exp3"] = dict(rows=rows, ref_unrelated=ref,
                   passed=bool(all(abs(r["function"] - 1) < 1e-9 and r["full"] < 0.999 and not r["identical_vec"] and r["full"] > ref for r in rows)))
print("PASS" if OUT["exp3"]["passed"] else "FAIL")

# ------------------------------------------------------------------ EXP 4: irrelevant capabilities
header("EXP 4  Irrelevant capabilities / goal relevance (state-dependent)")
lib_names = list(enc)
lib = [enc[n] for n in lib_names]
OUT["exp4"] = {}
for sname in ("warm", "cold"):
    st = emb.encode(p.states[sname])
    rel = emb.relevance_all(goal, st, lib)
    print(f"\n-- state '{sname}'  goal progress {emb.goal_progress(st, goal):.2f}")
    irr = set(p.irrelevant())
    for n, r in sorted(zip(lib_names, rel), key=lambda x: -x[1]):
        print(f"  {n:22s} relevance={r:.3f}  {'[irrelevant by design]' if n in irr else ''}")
    labels = [n not in irr for n in lib_names]
    # in the cold state Authenticate/CreateCart/AddItem are relevant; in warm they are redundant (already satisfied)
    redundant = {"Authenticate", "CreateCart", "AddItem"} if sname == "warm" else set()
    keep = [i for i, n in enumerate(lib_names) if n not in redundant]
    a = auc(rel[keep], np.array(labels)[keep])
    print(f"  AUC(relevance separates useful from irrelevant) = {a:.3f}")
    OUT["exp4"][sname] = dict(relevance=dict(zip(lib_names, map(float, rel))), auc=a)
# composable but irrelevant: GenerateReport follows CreateOrder yet does not serve the goal
cr = emb.compat(enc["CreateOrder_API"], enc["GenerateReport"])
print(f"\nCreateOrder_API -> GenerateReport composable={cr['composable']} but relevance="
      f"{OUT['exp4']['warm']['relevance']['GenerateReport']:.3f}  => composability != relevance")

# ------------------------------------------------------------------ EXP 5: operational attributes
header("EXP 5  Operational attributes (cost, reliability, availability, risk, resources)")
pays = ["MakePayment_Gateway", "MakePayment_Backup", "MakePayment_Bank"]
st = emb.encode(p.states["warm"])
# every payment variant is relevant to the goal equally -> differences come from operational blocks only
OUT["exp5"] = {}
for t in (12, 22):
    encs_t = {n: emb.encode_capability(p.caps[n], t=t) for n in pays}
    rel = emb.relevance_all(goal, st, [encs_t[n] for n in pays])
    print(f"\n-- at t={t}:00")
    OUT["exp5"][t] = {}
    for n, r in zip(pays, rel):
        c = encs_t[n]
        q = emb.blk(c, "Q")
        u = emb.utility(c, r)
        print(f"  {n:20s} rel={r:.2f} success_p={np.exp(-q[5]):.3f} unavailable={q[6]:.0f} risk={q[3]:.2f} "
              f"cost_scalar={emb.cost_scalar(c):.3f} resources={p.caps[n].resources}  utility={u:.4f}")
        OUT["exp5"][t][n] = u
fg = emb.similarity(enc["MakePayment_Gateway"], enc["MakePayment_Backup"], "function")
op = emb.similarity(enc["MakePayment_Gateway"], enc["MakePayment_Backup"], "operational")
print(f"\nGateway vs Backup: function similarity={fg:.3f} (same job), operational similarity={op:.3f} (different cost/risk)")
c_hi = emb.compose([enc["CreateOrder_API"], enc["MakePayment_Gateway"], enc["SendNotification"]])
print(f"composite reliability = product of parts = {np.exp(-emb.blk(c_hi, 'Q')[5]):.4f}")
c_bank = emb.compose([emb.encode_capability(p.caps["CreateOrder_API"], 22), emb.encode_capability(p.caps["MakePayment_Bank"], 22)])
print(f"chain containing Bank payment at 22:00 -> unavailable flag = {emb.blk(c_bank, 'Q')[6]:.0f} (any unavailable part blocks the composite)")
OUT["exp5"]["fn_sim"], OUT["exp5"]["op_sim"] = fg, op
OUT["exp5"]["bank_chain_unavail_22h"] = float(emb.blk(c_bank, "Q")[6])

# ------------------------------------------------------------------ EVALUATION across problems
header("EVALUATION  consistency / distinctness / composition validity / efficiency")
OUT["eval"] = {}
for name in ("ecommerce", "filepipeline", "assignment_pattern"):
    p, emb, enc = setup(name)
    names = list(enc)
    xs = [enc[n] for n in names]
    M = emb.compat_matrix(xs)
    n = len(names)
    off = ~np.eye(n, dtype=bool)
    truth = np.array([[symbolic_composable(p.caps[a], p.caps[b]) for b in names] for a in names])
    pred = M["composable"]
    agree = float((pred[off] == truth[off]).mean())
    cos = np.array([[emb.similarity(a, b, "full") for b in xs] for a in xs])
    fn = np.array([[emb.similarity(a, b, "function") for b in xs] for a in xs])
    res = dict(
        n_caps=n, dim=emb.dim, A=emb.A, D=emb.D,
        agree_with_symbolic=agree, auc_compat_score=auc(M["score"][off], truth[off]),
        auc_cosine_baseline=auc(cos[off], truth[off]), acc_cosine_baseline=best_acc(cos[off], truth[off]),
        n_composable_pairs=int(truth[off].sum()),
        distinct_vectors=len({tuple(np.round(x.vec, 9)) for x in xs}) == n,
        min_pair_distance=float(min(np.linalg.norm(a.vec - b.vec) for a, b in itertools.combinations(xs, 2))),
    )
    # enumerate all valid chains of length 2..4 and check vector-vs-formal and semantic soundness
    caps = [p.caps[k] for k in names]
    chains, stack = [], [[c] for c in caps]
    while stack and len(chains) < 400:
        ch = stack.pop()
        if len(ch) >= 2:
            chains.append(ch)
        if len(ch) < 4:
            acc = compose_formal(ch, strict=False)
            for nx in caps:
                if nx.name not in [c.name for c in ch] and symbolic_composable(acc, nx):
                    stack.append(ch + [nx])
    worst, sem_ok = 0.0, 0
    for ch in chains:
        cm = emb.compose([enc[c.name] for c in ch])
        worst = max(worst, float(np.abs(cm.vec - emb.encode(cm.obj).vec).max()))
        s0 = witness_state(cm.obj)                     # a state satisfying the composite's P/K/I
        s_end = run_chain(s0, ch)
        sem_ok += int(s_end is not None and s_end.satisfies(cm.obj.eff) and cm.obj.out_keys <= s_end.data)
    res.update(chains_tested=len(chains), max_vec_vs_formal_diff=worst, semantic_sound=sem_ok / max(len(chains), 1))
    t0 = time.perf_counter()
    for _ in range(200):
        emb.encode_capability(caps[0])
    res["encode_us"] = (time.perf_counter() - t0) / 200 * 1e6
    t0 = time.perf_counter()
    for _ in range(50):
        emb.compat_matrix(xs)
    res["compat_matrix_ms"] = (time.perf_counter() - t0) / 50 * 1e3
    res["bytes_per_cap"] = emb.dim * 8
    OUT["eval"][name] = res
    print(f"\n[{name}]")
    for k, v in res.items():
        print(f"  {k:26s} {v:.4g}" if isinstance(v, float) else f"  {k:26s} {v}")
    if name == "ecommerce":
        np.save("results/ecommerce_compat_score.npy", M["score"])
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
            for a_, mat, ttl in ((ax[0], M["score"], "compat score  C_i -> C_j  (ours)"), (ax[1], cos, "cosine similarity (full view)")):
                im = a_.imshow(mat, cmap="viridis", vmin=0, vmax=1)
                a_.set_xticks(range(n)); a_.set_yticks(range(n))
                a_.set_xticklabels(names, rotation=90, fontsize=7); a_.set_yticklabels(names, fontsize=7)
                a_.set_title(ttl); plt.colorbar(im, ax=a_, fraction=0.046)
            plt.tight_layout(); plt.savefig("results/compat_vs_cosine.png", dpi=130); plt.close()
        except Exception as e:  # noqa
            print("plot skipped:", e)

# synthetic scaling test
rng = random.Random(0)
from capemb.model import Capability  # noqa: E402
def synth(n):
    caps = []
    for i in range(n):
        v = rng.sample(range(60), 4)
        caps.append(Capability(name=f"c{i}", type="API", pre=[f"v{v[0]}==true"], eff=[f"v{v[1]}==true", f"v{v[2]}==true"],
                               inputs=[(f"d{rng.randrange(40)}", "X", "", True)], outputs=[(f"d{rng.randrange(40)}", "X", "")]))
    return caps
OUT["scaling"] = {}
for n in (100, 500, 2000):
    caps = synth(n)
    emb_s = Embedder(Vocabulary(caps))
    xs = [emb_s.encode_capability(c) for c in caps]
    t0 = time.perf_counter(); emb_s.compat_matrix(xs); dt = time.perf_counter() - t0
    OUT["scaling"][n] = dict(dim=emb_s.dim, all_pairs_ms=dt * 1e3, mem_MB=n * emb_s.dim * 8 / 1e6)
    print(f"synthetic n={n:5d} dim={emb_s.dim} all-pairs compat {dt*1e3:8.1f} ms  vectors {n*emb_s.dim*8/1e6:.2f} MB")

with open("results/results.json", "w") as fh:
    json.dump(OUT, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("\nsaved results/results.json")
