"""Exact speedups for PF-RPN (no retraining, same results).

RESULT OF THE FIRST GPU TEST: apply_fast_moe gives NO speedup. The router picks the biggest feature map (level 0) for
every test image, so the full O(N^2) pooling over ~16.7k tokens still has to run. Kept for reference; do not enable it.
apply_fast_predict is the useful one: PFRPN.predict loops `int(label)` over 900 GPU labels (900 host syncs, ~990 per
image in total, the main cause of ~105 ms GPU idle time). It only builds label names, so one .tolist() does the same.

--- original notes on the MoE rewrite ---
Exact speedup of PF-RPN's pseudo-text builder (no retraining, same maths).

Why: profiling shows _build_sparse_moe_pseudo_text takes ~35% of the time (162 of 467 ms). It runs
AttentionPool2d self-attention over ALL tokens of ALL 4 feature levels (the biggest map has ~16,700 tokens, so that
attention is O(N^2)), but then keeps only the top-k levels (k=2) for the meta_net. And the router only needs the
GLOBAL (first) token of each level, which is a single-query attention, O(N).

So: (1) global token for every level with a single query, (2) full pooled map only for the SELECTED levels.
The result is the same function. If the router selects the biggest level the saving is small; if it selects only
smaller levels the saving is large. Verify with bench/test_fast_moe.py before trusting it. UNTESTED on GPU here.
"""
import types
import torch
import torch.nn.functional as F


def global_token_only(pool, feat):
    """feat: [B, N, C] tokens of one level, in the order AttentionPool2d sees them. Returns [B, out_dim].
    Same as AttentionPool2d.forward(...)[0] but the attention has ONE query (the mean token)."""
    B, N, C = feat.shape
    x = feat.transpose(0, 1)                                   # [N, B, C]
    x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)     # [N+1, B, C]
    x = x + pool.positional_embedding[:N + 1][:, None, :]
    heads, hd = pool.num_heads, C // pool.num_heads
    q = F.linear(x[:1], pool.q_proj.weight, pool.q_proj.bias)  # [1, B, C]
    k = F.linear(x, pool.k_proj.weight, pool.k_proj.bias)      # [N+1, B, C]
    v = F.linear(x, pool.v_proj.weight, pool.v_proj.bias)
    q = q.reshape(1, B, heads, hd).permute(1, 2, 0, 3)         # [B, H, 1, hd]
    k = k.reshape(N + 1, B, heads, hd).permute(1, 2, 0, 3)     # [B, H, N+1, hd]
    v = v.reshape(N + 1, B, heads, hd).permute(1, 2, 0, 3)
    o = F.scaled_dot_product_attention(q, k, v)                # [B, H, 1, hd]
    o = o.permute(0, 2, 1, 3).reshape(B, C)
    return F.linear(o, pool.c_proj.weight, pool.c_proj.bias)


def _full_expert_feat(self, memory, offset, height, width):
    """The original per-level path: global + all local tokens -> [B, 1 + H*W, C]."""
    B = memory.shape[0]
    n = height * width
    vis = memory[:, offset:offset + n, :].reshape(B, height, width, -1).permute(0, 3, 1, 2).contiguous()
    g, local = self.vis_proj(vis)
    return torch.cat([g[:, :, None], local.flatten(2)], dim=2).permute(0, 2, 1).contiguous()


def fast_build_sparse_moe_pseudo_text(self, memory, spatial_shapes):
    B = memory.shape[0]
    shapes = spatial_shapes.tolist()
    offsets, off = [], 0
    for h, w in shapes:
        offsets.append(off)
        off += h * w
    globals_ = [global_token_only(self.vis_proj, memory[:, o:o + h * w, :]) for o, (h, w) in zip(offsets, shapes)]
    expert_feat = torch.stack(globals_, dim=1)
    router_weights = self.router(expert_feat)

    text_feat = self.learnable_text_embedding.expand(B, -1, -1).to(memory.device)
    k = min(self.topk, router_weights.size(1))
    idx_selected = torch.topk(router_weights, k=k, dim=1).indices.squeeze(-1)
    self.router._last_idx_selected = idx_selected.detach()
    weights_sel = torch.nn.Softmax(dim=1)(router_weights.gather(1, idx_selected.unsqueeze(-1))).squeeze(-1)
    meta_out = torch.zeros(B, idx_selected.size(1), text_feat.size(1), text_feat.size(2),
                           device=text_feat.device, dtype=text_feat.dtype)
    for expert_id in sorted(set(idx_selected.flatten().tolist())):      # one host sync for all levels
        b_idx, k_idx = (idx_selected == expert_id).nonzero(as_tuple=True)
        h, w = shapes[expert_id]
        feat = _full_expert_feat(self, memory, offsets[expert_id], h, w)
        meta_out[b_idx, k_idx] = self.meta_net(text_feat[b_idx], feat[b_idx]).to(meta_out.dtype)
    pseudo = (weights_sel[:, :, None, None] * meta_out).sum(dim=1)
    return pseudo, router_weights


def apply_fast_moe(model):
    """Replace the method on this model instance. Returns the original bound method so a test can compare."""
    original = model._build_sparse_moe_pseudo_text
    model._build_sparse_moe_pseudo_text = types.MethodType(fast_build_sparse_moe_pseudo_text, model)
    return original


def fast_predict(self, batch_inputs, batch_data_samples, rescale=True):
    """PFRPN.predict without the per-label host syncs. Same outputs, same label_names."""
    visual_feats = self.extract_feat(batch_inputs)
    text_dict = {}
    entity = ["object"]
    for data_samples in batch_data_samples:
        data_samples.token_positive_map = [1]
    head_inputs_dict = self.forward_transformer(visual_feats, text_dict, batch_data_samples)
    results_list = self.bbox_head.predict(**head_inputs_dict, rescale=rescale, batch_data_samples=batch_data_samples)
    for data_sample, pred_instances in zip(batch_data_samples, results_list):
        if len(pred_instances) > 0:
            pred_instances.label_names = [entity[i] if i < len(entity) else "unobject"
                                          for i in pred_instances.labels.tolist()]    # ONE sync
        data_sample.pred_instances = pred_instances
    return batch_data_samples


def apply_fast_predict(model):
    """Replace predict on this model instance. Returns the original bound method."""
    original = model.predict
    model.predict = types.MethodType(fast_predict, model)
    return original
