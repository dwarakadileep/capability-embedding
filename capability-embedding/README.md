# Capability Embedding for Composition (PCCST503 – Assignment 2)

A block-structured vector embedding of formally specified **states, goals and capabilities**.
Compatibility (`Ci -> Cj`), composition (`Cn o ... o C1`), goal relevance and operational trade-offs
are all computed with vector algebra, and are cross-checked against a symbolic ground truth.

## Layout
| Path | Purpose |
|---|---|
| `capemb/model.py` | formal model: atoms, states, goals, capabilities, symbolic composition, simulator |
| `capemb/embedding.py` | **the embedding**: `encode_*`, `compose`, `similarity`, `compat_matrix`, `relevance_all`, `utility` |
| `capemb/datasets.py` | experimental dataset (Deliverable 3); writes `data/*.json` |
| `experiments.py` | the 5 required experiments + evaluation + scaling |
| `tests/` | pytest suite (7 tests) |
| `results/` | `experiment_log.txt`, `results.json`, `compat_vs_cosine.png` |
| `REPORT.md` | technical report (Deliverables 1 and 4) |

## Run
```bash
pip install -r requirements.txt
python experiments.py        # prints every experiment, writes results/
python -m pytest -q
```

## API (Deliverable 2)
```python
from capemb import Embedder, Vocabulary, load
p   = load("ecommerce")
emb = Embedder(Vocabulary(list(p.caps.values()), [p.goal]))
s, g = emb.encode(p.states["warm"]), emb.encode(p.goal)          # encode(state), encode(goal)
c    = {n: emb.encode(x) for n, x in p.caps.items()}              # encode(capability)
comp = emb.compose([c["CreateOrder_API"], c["MakePayment_Gateway"], c["SendNotification"]])  # compose
emb.similarity(c["CreateOrder_API"], c["CreateOrder_DB"], view="function")   # similarity(x, y)
emb.compat(c["CreateOrder_API"], c["MakePayment_Gateway"])        # composability  Ci -> Cj
emb.relevance_all(g, s, list(c.values()))                         # goal relevance per capability
```

## Publish to GitHub
```bash
cd capability-embedding
git init && git add . && git commit -m "Capability embedding for composition"
git branch -M main
# create an empty repo named capability-embedding on github.com/dwarakadileep first, then:
git remote add origin https://github.com/dwarakadileep/capability-embedding.git
git push -u origin main
```
