import torch
import torch.nn.functional as F
import torch.nn as nn


def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))
    return torch.mean(loss)



def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.norm(emb, p=2)/emb.shape[0]
    return emb_loss * reg


def Stable_Adaptive_InfoNCE(view1, view2, temperature: float, b_cos: bool = True, alpha: float = 2.0):
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    sim_matrix = (view1 @ view2.T) / temperature  # (N, N)
    sim_matrix = sim_matrix + torch.eye(sim_matrix.size(0), device=sim_matrix.device) * alpha
    log_prob = F.log_softmax(sim_matrix, dim=1)  # (N, N)
    loss = -torch.diag(log_prob).mean()
    return loss


import math
import torch
import torch.nn.functional as F


def Stable_Adaptive_InfoNCE_Sampled(
    view1,
    view2,
    temperature: float,
    neg_samples: int,
    b_cos: bool = True,
    alpha: float = 2.0,
    sample_weight: torch.Tensor = None,
    mix_uniform: float = 0.5,
    strata_bins: int = 0,
    strata_uniform: float = 0.5,
    eps: float = 1e-12,
):
    assert temperature > 0, "temperature must be positive"

    if b_cos:
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)

    B = view1.size(0)
    neg_samples = int(neg_samples)

    # fall back to exact in-batch InfoNCE if K invalid
    if neg_samples <= 0 or neg_samples >= B:
        return Stable_Adaptive_InfoNCE(view1, view2, temperature, b_cos=False, alpha=alpha)

    # positive logit (diagonal)
    pos_logit = torch.sum(view1 * view2, dim=1) / temperature
    pos_logit = pos_logit + alpha  # match exact diagonal shift

    # ---- build proposal q in float32 for stable sampling ----
    device = view1.device
    if sample_weight is None:
        q = torch.full((B,), 1.0 / B, device=device, dtype=torch.float32)
        w = None
    else:
        w = sample_weight.to(device=device, dtype=torch.float32).clamp_min(0.0)
        w_sum = w.sum()
        if float(w_sum) <= 0.0:
            q_data = torch.full((B,), 1.0 / B, device=device, dtype=torch.float32)
        else:
            q_data = w / (w_sum + eps)
        q_uni = torch.full((B,), 1.0 / B, device=device, dtype=torch.float32)

        lam = float(mix_uniform)
        lam = 0.0 if lam < 0.0 else (1.0 if lam > 1.0 else lam)
        q = lam * q_uni + (1.0 - lam) * q_data
        q = q.clamp_min(1e-8)
        q = q / (q.sum() + eps)

    # ---- sample indices (optionally stratified) ----
    strata_bins = int(strata_bins)
    if (w is not None) and strata_bins and strata_bins > 1:
        # stratify by weight quantiles
        w_sorted, _ = torch.sort(w)
        cut_idx = torch.linspace(0, B, steps=strata_bins + 1, device=device).long()
        cut_idx[-1] = B
        cut_vals = w_sorted[cut_idx.clamp_max(B - 1)]

        idx_list = []
        remaining = neg_samples
        for b in range(strata_bins):
            low = cut_vals[b]
            high = cut_vals[b + 1]
            if b == strata_bins - 1:
                mask = (w >= low) & (w <= high)
            else:
                mask = (w >= low) & (w < high)
            cand_idx = torch.nonzero(mask, as_tuple=False).flatten()
            if cand_idx.numel() == 0:
                continue

            base_k = max(1, neg_samples // strata_bins)
            k = base_k if remaining >= base_k else remaining
            remaining -= k
            if k <= 0:
                continue

            q_cand = q.index_select(0, cand_idx)
            q_cand = q_cand / (q_cand.sum() + eps)

            uni_cand = torch.full_like(q_cand, 1.0 / q_cand.numel())
            lam_s = float(strata_uniform)
            lam_s = 0.0 if lam_s < 0.0 else (1.0 if lam_s > 1.0 else lam_s)
            q_cand = lam_s * uni_cand + (1.0 - lam_s) * q_cand
            q_cand = q_cand / (q_cand.sum() + eps)

            picked = torch.multinomial(q_cand, k, replacement=True)
            idx_list.append(cand_idx.index_select(0, picked))

        if remaining > 0:
            tail = torch.multinomial(q, remaining, replacement=True)
            idx_list.append(tail)

        idx = torch.cat(idx_list, dim=0) if len(idx_list) > 0 else torch.multinomial(q, neg_samples, replacement=True)
    else:
        idx = torch.multinomial(q, neg_samples, replacement=True)  # [K]

    K = int(idx.numel())
    if K <= 0:
        return Stable_Adaptive_InfoNCE(view1, view2, temperature, b_cos=False, alpha=alpha)

    # ---- negatives 
    neg_bank = view2.index_select(0, idx)  # [K, D]
    neg_logits = (view1 @ neg_bank.t()) / temperature  # [B, K]

    log_q = torch.log(q.index_select(0, idx) + eps).to(dtype=neg_logits.dtype)  # [K]

    logZ_neg = torch.logsumexp(neg_logits - log_q.view(1, -1), dim=1) - math.log(K)
    log_denom = torch.logaddexp(pos_logit, logZ_neg)

    loss = -(pos_logit - log_denom).mean()
    return loss
