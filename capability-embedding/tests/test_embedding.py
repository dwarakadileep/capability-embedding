import numpy as np, pytest
from capemb import Embedder, Vocabulary, load, CompositionError, symbolic_composable


def setup(name="ecommerce"):
    p = load(name)
    e = Embedder(Vocabulary(list(p.caps.values()), [p.goal]))
    return p, e, {n: e.encode(c) for n, c in p.caps.items()}


def test_pattern_compatibility():
    p, e, enc = setup("assignment_pattern")
    assert e.compat(enc["CreateOrder"], enc["MakePayment"])["composable"]
    r = e.compat(enc["CreateOrder"], enc["CancelCart"])
    assert not r["composable"] and r["conflict"] > 0


def test_vector_compose_equals_formal_encoding():
    p, e, enc = setup()
    for chain in p.chains:
        c = e.compose([enc[n] for n in chain])
        assert np.allclose(c.vec, e.encode(c.obj).vec)


def test_associativity():
    p, e, enc = setup()
    a, b, c = (enc[n].vec for n in p.chains[0])
    assert np.allclose(e.compose_vectors([e.compose_vectors([a, b]), c]), e.compose_vectors([a, e.compose_vectors([b, c])]))


def test_invalid_order_rejected():
    p, e, enc = setup()
    with pytest.raises(CompositionError):
        e.compose([enc[n] for n in reversed(p.chains[0])])


def test_alternatives_similar_not_identical():
    p, e, enc = setup()
    a, d = enc["CreateOrder_API"], enc["CreateOrder_DB"]
    assert e.similarity(a, d, "function") == pytest.approx(1.0)
    assert e.similarity(a, d, "full") < 0.999 and not np.allclose(a.vec, d.vec)
    assert not e.compat(a, d)["composable"]          # similar != composable


def test_matrix_matches_symbolic_oracle():
    for name in ("ecommerce", "filepipeline"):
        p, e, enc = setup(name)
        names = list(enc)
        M = e.compat_matrix([enc[n] for n in names])["composable"]
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i != j:
                    assert M[i, j] == symbolic_composable(p.caps[a], p.caps[b])


def test_state_dependent_relevance_and_availability():
    p, e, enc = setup()
    goal, lib = e.encode(p.goal), list(enc.values())
    warm = e.relevance_all(goal, e.encode(p.states["warm"]), lib)
    cold = e.relevance_all(goal, e.encode(p.states["cold"]), lib)
    names = list(enc)
    assert warm[names.index("Authenticate")] == 0 < cold[names.index("Authenticate")]
    assert warm[names.index("GenerateReport")] == 0
    bank = e.encode_capability(p.caps["MakePayment_Bank"], t=22)
    assert e.utility(bank, 1.0) == 0.0
