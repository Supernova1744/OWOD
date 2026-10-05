"""The maths behind owod/pfrpn_fast.global_token_only, checked with numpy only:
attention output for the FIRST token (one query) equals row 0 of full self-attention."""
import numpy as np


def mha(q_in, kv_in, Wq, bq, Wk, bk, Wv, bv, Wo, bo, heads):
    L, C = q_in.shape
    hd = C // heads
    q = (q_in @ Wq.T + bq).reshape(L, heads, hd).transpose(1, 0, 2)
    k = (kv_in @ Wk.T + bk).reshape(kv_in.shape[0], heads, hd).transpose(1, 0, 2)
    v = (kv_in @ Wv.T + bv).reshape(kv_in.shape[0], heads, hd).transpose(1, 0, 2)
    s = q @ k.transpose(0, 2, 1) / np.sqrt(hd)
    s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
    o = (s @ v).transpose(1, 0, 2).reshape(L, C)
    return o @ Wo.T + bo


def test_single_query_equals_row0_of_full_attention():
    rng = np.random.default_rng(0)
    N, C, heads = 37, 64, 8
    P = lambda *s: rng.standard_normal(s) * 0.1
    Wq, Wk, Wv, Wo = P(C, C), P(C, C), P(C, C), P(C, C)
    bq, bk, bv, bo = P(C), P(C), P(C), P(C)
    tokens = rng.standard_normal((N, C))
    x = np.concatenate([tokens.mean(0, keepdims=True), tokens]) + P(N + 1, C)
    full = mha(x, x, Wq, bq, Wk, bk, Wv, bv, Wo, bo, heads)
    one = mha(x[:1], x, Wq, bq, Wk, bk, Wv, bv, Wo, bo, heads)
    assert np.allclose(full[0], one[0], atol=1e-10)
    assert not np.allclose(full[1], full[0])      # sanity: other rows differ
