# A Block-Structured Vector Embedding for Capability Composition

*PCCST503 – Assignment 2. All numbers below are produced by `python experiments.py` (see `results/experiment_log.txt`).*

## 1. Problem definition
Given a formal application `A = (S, C, S_I, G, R, K)`, represent states, goals and capabilities
`C_i = (T, I, O, P, E, K, R, Q, Rel, A, M)` as vectors so that the vector space preserves
(i) capability identity, (ii) which states a capability can run in, (iii) which capabilities can follow
which (`E_i ⇒ P_j`, `O_i ⊨ I_j`), (iv) how capabilities compose, (v) goal relevance, and
(vi) operational trade-offs. The planner of Assignment 1 is out of scope.

## 2. Design requirements
| # | Requirement | Design consequence |
|---|---|---|
| R1 | Similar ≠ composable | two separate relations: *similarity* (symmetric, cosine) and *compatibility* (asymmetric, directional) |
| R2 | Composition must be meaningful for chains | a closed, associative operator on vectors |
| R3 | Compatibility must respect logical entailment (`x==5 ⇒ x>0`) and contradiction (`x==true` vs `x==false`) | atoms + entailment/conflict matrices |
| R4 | Implementation ≠ function | mechanism/type kept in separate low-weight blocks |
| R5 | Operational properties must not corrupt semantics | separate `Q` block, used only in utility and an "operational" view |
| R6 | State/goal comparable with capabilities | same atom vocabulary for all three entity kinds |

## 3. Related embedding approaches
* **Word2Vec / sentence encoders** capture distributional similarity. Two capabilities that *do the same thing* are close, but so are `CreateOrder` and `CancelOrder`; and a producer/consumer pair like `CreateOrder → MakePayment` is *not* close. Similarity is symmetric, composability is not. (Our baseline in §9 confirms this.)
* **Knowledge-graph embeddings (TransE-style)** model a relation as a translation `h + r ≈ t`; this suggests representing a capability as a state→state translation, but they are learned from data, unreliable for exact logical conditions, and give no compositional guarantee.
* **Order/box embeddings** represent entailment geometrically; our `Ent` matrix is a discrete version of this for conditions.
* **STRIPS/PDDL regression** gives the exact symbolic composition rule `P12 = P1 ∪ (P2 \ E1)`, `E12 = E2 ∪ (E1 \ overwritten)`; we *vectorise* it.
* **Vector-symbolic / hyperdimensional computing** uses binding and bundling for structured data; our blocks use bundling (sums/maxima) over an explicit vocabulary instead of random hypervectors, which keeps every dimension interpretable.

*Limitations of existing methods for this task*: learned embeddings lack guarantees; purely symbolic methods have no notion of graded similarity, goal relevance in a metric space, or compact batch computation. The design below keeps the guarantees of the symbolic model while living in a vector space.

## 4. Proposed representation
```
phi_C(C) = [ P | E | K | I | O | T | M | R | Q | N ]
```
| Block | Dim | Content |
|---|---|---|
| P, E, K | A each | indicator over the vocabulary of atomic conditions (preconditions, effects, constraints) |
| I | D | required inputs = 1.0, optional = 0.5, over data items `name:type` |
| O | D | produced outputs |
| T | 9 | mean one-hot of capability type |
| M | 16 | feature-hashed execution mechanism (signed hashing, L2-normalised, mean over parts) |
| R | \|R\| | multi-hot resources |
| Q | 7 | `[time_ms, money, resource, risk, energy, −ln Rel, 1−Avail(t)]` |
| N | 1 | number of atomic parts |

* **State** `phi_S(S) = [holds(a_1..a_A) | has(d_1..d_D)]` – every vocabulary atom evaluated against `S`.
* **Goal** `phi_G(G)` = indicator over its atoms. Because states, goals and capabilities share the atom axis, they are directly comparable.
* **Design decisions** (assignment asks for them explicitly):
  * *Type & mechanism* sit in low-weight blocks (0.5 in the `full` view, 0 in `function`), so API/DB/GUI variants of one function are close but **not identical**.
  * *Reliability* is stored as `−ln Rel` so reliabilities of a chain **add** (product of probabilities). *Availability* is a time-dependent flag `Avail(t)` evaluated at encode time.
  * *Cost, reliability, availability, risk* live in the **separate Q block**: they never influence compatibility, they enter only `utility` and the `operational` similarity view. *Resources* are in a block of their own (R) with small weight in `full`.
  * *Constraints* K are a block parallel to P (checked by `applicability`, unioned by composition).

## 5. Mathematical formulation
Let atoms `a = (var, op, val)`; `a ⊢ b` ("a entails b") defines `Ent ∈ {0,1}^{A×A}`; `X[a,b]=1` iff `a` is `var==v` and contradicts `b` on the same variable; `V ∈ {0,1}^{A×nvars}` maps atoms to variables.
Effect closure: `Ē = clip(E · Ent, 0, 1)`.

**Directional compatibility.** With provision `π_i = [Ē_i ‖ O_i]` and requirement `ρ_j = [P_j ‖ I_j]`:
```
feed(i→j)     = π_i · ρ_j / ‖ρ_j‖₁                    ∈ [0,1]
conflict(i→j) = clip(E_i X, 0, 1) · P_j / ‖P_j‖₁       ∈ [0,1]
score(i→j)    = feed · (1 − conflict)
composable(i→j) ⇔ feed > 0 ∧ conflict = 0
```
`feed_pre` and `feed_io` report the state-condition and data channels separately. For all pairs at once: `feed = Π Rᵀ` (one matrix product).

**Similarity.** `sim_view(x,y) = cos(W_view ⊙ φ(x), W_view ⊙ φ(y))`, with views
`function` (E,O only: what it *does*), `interface` (P,E,K,I,O), `full` (+T 0.5, M 0.5, R 0.25),
and `operational` = `1 − ‖q̂_x − q̂_y‖/√7` on normalised Q. State–goal similarity is goal progress `g·s/|g|`; state–capability similarity is applicability `(P·s_a + K·s_a + I·s_d)/(|P|+|K|+|I|)`.

**Composition** `C₂ ∘ C₁` (vector operator `⊗`, applied left to right; `N=n₁+n₂`):
```
P  = max( P1 , P2 ⊙ (1 − Ē1) )               unmet preconditions bubble up
E  = max( E2 , E1 ⊙ (1 − clip((E2 V)Vᵀ,0,1)) ) later effect on a variable overwrites earlier
K  = max(K1,K2)                  R = max(R1,R2)          O = max(O1,O2)
I  = max( I1 , I2 ⊙ (1 − O1) )               inputs produced earlier vanish
T,M = (n1·T1 + n2·T2)/(n1+n2)                size-weighted mean
Q: time, money, resource, energy, −ln Rel add;  risk = 1−(1−r1)(1−r2);  unavailable = max
```
Properties (verified numerically): **closed** (the result is again a capability vector, usable in `compat`, `similarity`, and further composition), **associative** (gap ≈ 1e-16), and **homomorphic**: `compose(φ(C1),…,φ(Cn)) = φ(formal_compose(C1..Cn))` (max error 1.7e-16 over 400 chains).

**Goal relevance** is backward need-propagation in vector space. `need₀ = g ⊙ (1 − s_a)`; `score_k(c) = π_c·need_k/‖need_k‖₁`; `need_{k+1} = ⋃_{c: score_k(c)>0} ρ_c ⊙ (1−s)`; `relevance(c) = max_k γ^k score_k(c)` (γ=0.5, depth 3). It is **state-aware**: conditions already true in `s` are not needed.

**Utility** `u(c) = relevance · e^{−q_Rel} · (1−unavail) / (1 + w·q_{0..4})`.

## 6. Capability composition model
A composite is a capability with `type = COMPOSITE`, `N>1`, the *external* interface of the chain (its P, I are what the chain needs from outside; its E, O are what it ends up providing) and aggregated Q. Validity is a property of the chain: each step must be linked to the composite built so far (so `C3` may be fed by `C1`) and must not conflict with it; reversed orders are rejected (`no link SendNotification -> MakePayment_Gateway`). Vector-level relationship to the parts: the composite's cosine to its parts is highest for the first part (it contributes the P and I blocks), and falls for later parts (their P and I are internalised).

## 7. Implementation
`capemb/` (≈600 lines). API: `encode(state|goal|capability)`, `compose([caps])`, `similarity(x,y,view)`, plus `compat`, `compat_matrix`, `applicability`, `goal_progress`, `relevance_all`, `utility`. Dataset in `data/*.json`; tests in `tests/`.

## 8. Experimental methodology
Three formally specified problems: **ecommerce** (15 capabilities incl. 3 CreateOrder implementations, 3 payment variants, 5 irrelevant ones, 2 initial states, 3 reference chains), **filepipeline** (8 capabilities incl. conflicting distractors `ResizeImage`, `DeleteFile`), and **assignment_pattern** (the exact C1/C2/C3 of Experiment 1). Ground truth is a symbolic oracle (`symbolic_composable`) and a state simulator. Baselines: cosine similarity as a predictor of composability. Same code and same hyper-parameters are used for all problems.

## 9. Results
**Exp 1 – compatibility.** `CreateOrder→MakePayment`: feed 1.0, conflict 0, composable. `CreateOrder→CancelCart`: feed 0, conflict 1, not composable. Cosine (full) is 0.194 vs 0.065 – a tiny gap that says nothing about direction or conflict.

**Exp 2 – composition** `CreateOrder_API → MakePayment_Gateway → SendNotification`: composite pre = {authenticated, cart exists, item_count>0, inventory available}; inputs {cart_id, payment_method} (order_id vanished, produced inside); effects {Order.exists, Order.status, Cart.locked, Payment.status=SUCCESS, Notification.sent}; reliability 0.931 = 0.99·0.95·0.99; time 980 ms. Cosine to parts (function view): 0.756 / 0.535 / 0.378. Executing the chain from the warm state raises goal progress 0.00 → 1.00.

**Exp 3 – alternatives** (API / DB / GUI): function-similarity = 1.000 for all pairs, full similarity 0.947–0.950 (vectors not identical), versus 0.052 for an unrelated capability; they are *not* composable with each other (similar ≠ composable).

**Exp 4 – irrelevant capabilities.** Warm state: the 7 goal-serving capabilities score 0.333, the 5 irrelevant ones 0.000 (AUC 1.0). Cold state: `CreateCart` 0.143, `Authenticate` / `AddItem` 0.071 appear through 2nd-order need propagation, they were 0 in the warm state (state-awareness). `GenerateReport` is composable after `CreateOrder` yet has relevance 0 → composability ≠ relevance.

**Exp 5 – operational attributes.** The three payment variants have identical relevance; utility at 12:00 ranks Bank 0.116 > Gateway 0.099 > Backup 0.073 (cost, risk, reliability, resources); at 22:00 the Bank variant (window 09–17) drops to 0.000 and any chain containing it gets `unavailable = 1`. Gateway vs Backup: function-similarity 1.000 but operational similarity 0.958.

**Evaluation table**

| Property | ecommerce | filepipeline | pattern |
|---|---|---|---|
| Distinct vectors | yes | yes | yes |
| Compat agrees with symbolic oracle (all ordered pairs) | 100 % | 100 % | 100 % |
| AUC compat score / **cosine baseline** | 1.00 / **0.45** | 1.00 / **0.42** | 1.00 / 0.90 |
| Vector-compose vs formal (max error), #chains | 1.7e-16, 400 | 1.2e-16, 91 | 0, 1 |
| Chains semantically sound (simulated) | 100 % | 100 % | 100 % |
| Dim / bytes per capability | 114 / 912 | 76 / 608 | 45 / 360 |
| encode (µs) / all-pairs compat (ms) | 17 / 0.12 | 17 / 0.08 | 14 / 0.07 |

Scaling (synthetic): 2 000 capabilities, dim 293 → all-pairs compatibility in ≈150 ms, 4.7 MB.

## 10. Analysis
* **Why similarity is not enough**: the cosine baseline has AUC ≈ 0.45 for composability on the two realistic problems (below chance) because alternative implementations are near-identical but never feed each other, while producer/consumer pairs are dissimilar. Compatibility must be directional and based on `E/O → P/I`.
* **Composition** is the key positive result: a single closed, associative operator whose output is exactly the encoding of the formally composed capability, and whose chain semantics were validated by simulation.
* **Honest caveat**: 100 % agreement with the symbolic oracle is *by design* – the vector operations implement the same logic in matrix form. The value is that (i) it is vectorised, graded (`feed`, `conflict`) and comparable to states/goals, (ii) composites live in the same space, (iii) the baseline comparison shows the structure is necessary.
* **Partial links**: `CreateOrder_API → SendNotification` has `feed_io = 1` (order_id) but `feed_pre = 0` (needs payment). The channels are therefore reported separately instead of hiding a partial dependency.
* **Operational properties** deliberately do not alter semantic similarity; they are a separate view and enter ranking only through `utility`.

## 11. Limitations
1. Vocabulary is fixed per problem (dimension grows with the number of distinct conditions); new conditions need re-indexing. Hashing atoms would avoid this at the cost of collisions.
2. Atoms are single-variable comparisons against constants; relations between variables (`quantity <= inventory`) must be reified as boolean atoms.
3. Entailment is only between atoms on the same variable (no multi-variable reasoning).
4. Compatibility is exact/logical, not learned; it cannot generalise from natural-language descriptions.
5. Relevance is a bounded-depth heuristic, not planning; Q aggregation assumes independence and sequential execution; utility weights are hand-set.
6. Dataset is small and hand-built (26 capabilities across the three problems).

## 12. Conclusion
A block-structured embedding that shares one atom axis across states, goals and capabilities can represent identity, state applicability, precondition–effect and input–output compatibility, goal relevance and operational trade-offs, and supports an associative composition operator that is exact with respect to the formal composition. Similarity and composability are cleanly separated: similarity answers "does it do the same thing", compatibility answers "can it follow". Future work: learned/hashed vocabularies, soft (probabilistic) conditions, learning `w` from execution logs, and combining the structured vector with a text-embedding block.
