import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from Params import args
from mamba_ssm import Mamba
from Utils.loss_torch import (
    bpr_loss,
    l2_reg_loss,
    Stable_Adaptive_InfoNCE,
    Stable_Adaptive_InfoNCE_Sampled
)


class GNNLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, adj, embeds):
        return torch.sparse.mm(adj, embeds)


class MambaDepthGating(nn.Module):
    """
    Degree-conditioned Mamba depth gating:
      - logits: [B, L]
      - fused : [B, D]

    Modified:
      - remove activation dropout on y
      - add new view augmentations specialized for L=2 (alpha-gating)
        (only active when perturbed=True, used for CL view)
    """
    def __init__(
        self,
        d_model,
        gate_dim=16,
        d_state=8,
        d_conv=4,
        expand=2,
        dropout=0.0,     # kept for compatibility; unused
        temp=1.0,
        depth_drop=0.0,
        depth_noise=0.0,
        deg_cond=True,

        # ===== NEW: L=2 specialized view aug knobs =====
        beta_k=0.0,              # args.gate_beta_k (e.g., 10~80). 0 disables
        concrete_tau=0.0,        # args.gate_concrete_tau (e.g., 0.3~1.0). 0 disables
        t_df=0.0,                # args.gate_t_df (e.g., 3,5,8). 0 disables
        t_scale=0.0,             # args.gate_t_scale (e.g., 0.02~0.10). 0 disables
        adv_eps=0.0,             # args.gate_adv_eps (FGSM on diff; optional). 0 disables
    ):
        super().__init__()
        self.gate_dim = int(gate_dim)
        self.temp = float(temp)
        self.depth_drop = float(depth_drop)
        self.depth_noise = float(depth_noise)
        self.deg_cond = bool(deg_cond)

        # new aug params
        self.beta_k = float(beta_k)
        self.concrete_tau = float(concrete_tau)
        self.t_df = float(t_df)
        self.t_scale = float(t_scale)
        self.adv_eps = float(adv_eps)

        self.down = nn.Linear(d_model, self.gate_dim, bias=False)
        self.mamba = Mamba(d_model=self.gate_dim, d_state=d_state, d_conv=d_conv, expand=expand)

        if self.deg_cond:
            self.deg_proj = nn.Linear(1, self.gate_dim, bias=True)
            nn.init.normal_(self.deg_proj.weight, std=0.02)
            nn.init.zeros_(self.deg_proj.bias)

        self.ln = nn.LayerNorm(self.gate_dim, eps=1e-12)
        self.to_logit = nn.Linear(self.gate_dim, 1, bias=True)

        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.normal_(self.to_logit.weight, std=0.02)
        nn.init.zeros_(self.to_logit.bias)

    def _compute_logits(self, seq, deg=None):
        g = self.down(seq)  # [B, L, gate_dim]

        if self.deg_cond and (deg is not None):
            if deg.dim() == 1:
                deg = deg.unsqueeze(-1)  # [B,1]
            deg = deg.to(device=seq.device, dtype=g.dtype)
            deg_bias = self.deg_proj(deg).unsqueeze(1)  # [B,1,gate_dim]
            g = g + deg_bias
        #y = self.ln(g)
        y = self.mamba(g)      # [B, L, gate_dim]
        y = self.ln(y + g)
        logits = self.to_logit(y).squeeze(-1)  # [B, L]
        return logits

    def _augment_logits_general(self, logits):
        """
        keep your existing behavior:
          - depth_drop for L>2 mid tokens (unused when L=2)
          - gaussian depth_noise (must keep)
        """
        B, L = logits.shape
        out = logits

        if self.depth_drop > 0 and L > 2:
            keep = torch.ones(B, L, device=logits.device, dtype=torch.bool)
            drop_mid = (torch.rand(B, L - 2, device=logits.device) < self.depth_drop)
            keep[:, 1:-1] = ~drop_mid
            neg_inf = torch.finfo(out.dtype).min
            out = out.masked_fill(~keep, neg_inf)

        # MUST keep gaussian noise (your fixed depth_noise=0.05)
        if self.depth_noise > 0:
            out = out + self.depth_noise * torch.randn_like(out)

        return out

    def _alpha_from_logits_L2(self, logits):
        """
        logits: [B,2]
        return alpha in (0,1): w = [1-alpha, alpha]
        """
        temp = max(self.temp, 1e-6)
        diff = (logits[:, 1] - logits[:, 0]) / temp   # [B]
        alpha = torch.sigmoid(diff)                   # [B]
        return alpha, diff

    def _sample_alpha_beta(self, alpha):
        """
        alpha: [B] in (0,1)
        Beta(k*alpha, k*(1-alpha)) centered at alpha; rsample for gradient.
        """
        k = self.beta_k
        if k <= 0:
            return alpha
        eps = 1e-4
        a = (alpha * k).clamp_min(eps)
        b = ((1.0 - alpha) * k).clamp_min(eps)
        dist = torch.distributions.Beta(a, b)
        # rsample enables reparameterization gradient
        return dist.rsample().clamp(0.0, 1.0)

    def _sample_alpha_concrete(self, diff):
        """
        diff: [B] where alpha = sigmoid(diff)
        sample alpha via RelaxedBernoulli(logits=diff)
        """
        tau = self.concrete_tau
        if tau <= 0:
            return None
        dist = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
            temperature=torch.tensor(tau, device=diff.device, dtype=diff.dtype),
            logits=diff
        )
        return dist.rsample().clamp(0.0, 1.0)

    def _heavy_tail_diff(self, diff):
        """
        Student-t heavy-tail noise on diff (L=2).
        """
        if (self.t_df <= 0) or (self.t_scale <= 0) or (not self.training):
            return diff
        df = torch.tensor(self.t_df, device=diff.device, dtype=diff.dtype)
        dist = torch.distributions.StudentT(df=df, loc=torch.zeros_like(diff), scale=torch.ones_like(diff))
        noise = dist.rsample() * self.t_scale
        return diff + noise

    def fuse_from_logits(self, seq, logits, perturbed=False, adv_grad=None):
        """
        seq:   [B, L, D]
        logits:[B, L]
        perturbed: used for CL view
        adv_grad: optional adversarial gradient on diff (L=2), shape [B]
        """
        # --- apply baseline logit aug (kept) ---
        if perturbed:
            logits = self._augment_logits_general(logits)

        B, L, D = seq.shape

        # ===== L=2 specialized alpha view =====
        if L == 2:
            alpha, diff = self._alpha_from_logits_L2(logits)

            # optional heavy-tail on diff (only for perturbed view)
            if perturbed:
                diff = self._heavy_tail_diff(diff)
                alpha = torch.sigmoid(diff)

                # optional FGSM on diff if adv_grad is provided
                if (self.adv_eps > 0) and (adv_grad is not None):
                    # adv_grad should be normalized outside if desired
                    diff = diff + self.adv_eps * adv_grad
                    alpha = torch.sigmoid(diff)

                # Beta-centered sampling (centered at alpha)
                if self.beta_k > 0:
                    alpha = self._sample_alpha_beta(alpha)

                # Concrete sampling (directly from diff)
                if self.concrete_tau > 0:
                    alpha_c = self._sample_alpha_concrete(diff)
                    if alpha_c is not None:
                        alpha = alpha_c

            # fuse
            alpha = alpha.view(B, 1)  # [B,1]
            fused = (1.0 - alpha) * seq[:, 0, :] + alpha * seq[:, 1, :]
            return fused

        # ===== general L>2 =====
        w = torch.softmax(logits / max(self.temp, 1e-6), dim=1)  # [B, L]
        fused = torch.sum(seq * w.unsqueeze(-1), dim=1)
        return fused

    def forward(self, seq, deg=None, perturbed=False, return_logits=False):
        logits = self._compute_logits(seq, deg=deg)
        fused = self.fuse_from_logits(seq, logits, perturbed=perturbed)
        if return_logits:
            return fused, logits
        return fused


class LMGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.user_embedding = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(args.user, args.latdim))
        )
        self.item_embedding = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(args.item, args.latdim))
        )

        self.num_gnn_layers = int(getattr(args, "num_gnn_layers", 0))
        self.gnn_layers = nn.ModuleList([GNNLayer() for _ in range(self.num_gnn_layers)])

        # ===== gating hyperparams (keep) =====
        self.gate_dim = int(getattr(args, "gate_dim", 16))
        self.gate_temp = float(getattr(args, "gate_temp", 0.8))      # keep 0.8
        self.depth_drop = float(getattr(args, "depth_drop", 0.1))
        self.depth_noise = float(getattr(args, "depth_noise", 0.0)) # keep 0.05

        # degree conditioning
        self.deg_cond = bool(int(getattr(args, "deg_cond", 0)))
        self.deg_norm = bool(int(getattr(args, "deg_norm", 0)))

        # ===== NEW aug args =====
        self.gate_beta_k = float(getattr(args, "gate_beta_k", 0.0))
        self.gate_concrete_tau = float(getattr(args, "gate_concrete_tau", 0.0))
        self.gate_t_df = float(getattr(args, "gate_t_df", 0.0))
        self.gate_t_scale = float(getattr(args, "gate_t_scale", 0.0))
        self.gate_adv_eps = float(getattr(args, "gate_adv_eps", 0.0))  # optional

        self.depth_gater = MambaDepthGating(
            d_model=args.latdim,
            gate_dim=self.gate_dim,
            d_state=int(getattr(args, "d_state", 8)),
            d_conv=int(getattr(args, "d_conv", 4)),
            expand=int(getattr(args, "expand", 2)),
            dropout=float(getattr(args, "gate_dropout", 0.0)),   # unused
            temp=self.gate_temp,
            depth_drop=self.depth_drop,
            depth_noise=self.depth_noise,
            deg_cond=self.deg_cond,

            beta_k=self.gate_beta_k,
            concrete_tau=self.gate_concrete_tau,
            t_df=self.gate_t_df,
            t_scale=self.gate_t_scale,
            adv_eps=self.gate_adv_eps,
        )

        # losses (keep)
        self.cl_rate = float(getattr(args, "lambda_cl", 0.0))  # keep 0.04
        self.temp = float(getattr(args, "tau", 0.2))
        self.reg = float(getattr(args, "reg", 1e-4))

        self.register_buffer("node_logdeg", torch.empty(0), persistent=False)

    @torch.no_grad()
    def set_node_degree(self, user_deg: torch.Tensor, item_deg: torch.Tensor):
        deg = torch.cat([user_deg, item_deg], dim=0).float()
        deg = torch.clamp(deg, min=0.0)
        logdeg = torch.log1p(deg)

        if self.deg_norm:
            mean = logdeg.mean()
            std = logdeg.std().clamp_min(1e-6)
            logdeg = (logdeg - mean) / std

        self.node_logdeg = logdeg.to(device=args.device)

    def _get_deg_for_ids(self, node_ids: torch.Tensor):
        if (not self.deg_cond) or (self.node_logdeg.numel() == 0):
            return None
        return self.node_logdeg.index_select(0, node_ids.long())

    def _build_depth_seqs_for_ids(self, adj, node_ids):
        node_ids = node_ids.long()
        embeds = torch.cat([self.user_embedding, self.item_embedding], dim=0)  # [N, D]

        seq_list = [embeds.index_select(0, node_ids)]
        for l in range(self.num_gnn_layers):
            embeds = self.gnn_layers[l](adj, embeds)
            embeds = F.normalize(embeds, p=2, dim=1)
            seq_list.append(embeds.index_select(0, node_ids))

        seq = torch.stack(seq_list, dim=1)  # [M, L, D]
        return seq

    def calculate_loss(
        self,
        user_emb,
        pos_emb,
        neg_emb,
        aug_user_emb=None,
        aug_item_emb=None,
        user_weight=None,
        item_weight=None,
    ):
        rec_loss = bpr_loss(user_emb, pos_emb, neg_emb)
        reg_loss = l2_reg_loss(self.reg, user_emb, pos_emb, neg_emb)

        if (aug_user_emb is not None) and (aug_item_emb is not None) and (self.cl_rate > 0):
            v1 = torch.cat([user_emb, pos_emb], dim=0)           # [2B, D]
            v2 = torch.cat([aug_user_emb, aug_item_emb], dim=0)  # [2B, D]

            if (user_weight is None) and (item_weight is None):
                w = None
            else:
                B = user_emb.size(0)
                device = user_emb.device
                dtype = user_emb.dtype
                uw = user_weight.to(device=device, dtype=dtype) if user_weight is not None else torch.ones(B, device=device, dtype=dtype)
                iw = item_weight.to(device=device, dtype=dtype) if item_weight is not None else torch.ones(B, device=device, dtype=dtype)
                w = torch.cat([uw, iw], dim=0)  # [2B]

            cl_all = Stable_Adaptive_InfoNCE_Sampled(
                v1, v2,
                self.temp,
                args.cl_neg_samples,
                b_cos=True,
                alpha=2.0,
                sample_weight=w,
                mix_uniform=args.cl_mix_uniform,
                strata_bins=args.cl_strata_bins,
                strata_uniform=args.cl_strata_uniform,
            )

            cl_loss = self.cl_rate * (2.0 * cl_all)
            total_loss = rec_loss + reg_loss + cl_loss
        else:
            cl_loss = torch.tensor(0.0, device=user_emb.device)
            total_loss = rec_loss + reg_loss

        return total_loss, rec_loss, cl_loss

    def calcLosses(self, ancs, poss, negs, adj, user_weight=None, item_weight=None):
        U = int(args.user)
        ancs = ancs.long()
        poss = poss.long()
        negs = negs.long()

        pos_g = poss + U
        neg_g = negs + U
        B = ancs.shape[0]

        node_ids = torch.cat([ancs, pos_g, neg_g], dim=0)  # [3B]
        seq_all = self._build_depth_seqs_for_ids(adj, node_ids)      # [3B, L, D]
        deg_all = self._get_deg_for_ids(node_ids)                    # [3B] or None

        fused_all, logits_all = self.depth_gater(
            seq_all, deg=deg_all, perturbed=False, return_logits=True
        )

        user_emb = fused_all[:B]
        pos_emb  = fused_all[B:2 * B]
        neg_emb  = fused_all[2 * B:]

        # aug view: reuse logits; apply new alpha-based augmentation
        if self.cl_rate > 0:
            seq_up = seq_all[:2 * B]          # [2B, L, D]
            logits_up = logits_all[:2 * B]    # [2B, L]

            # optional: adversarial on diff (FGSM) — stronger but costs 1 autograd.grad
            adv_grad = None
            if getattr(args, "gate_adv_eps", 0.0) > 0 and seq_up.size(1) == 2:
                logits_req = logits_up.detach().requires_grad_(True)
                fused_tmp = self.depth_gater.fuse_from_logits(seq_up, logits_req, perturbed=True)
                # proxy loss: push away from clean (stop-grad) to get informative direction
                clean = fused_all[:2 * B].detach()
                proxy = (1.0 - F.cosine_similarity(clean, fused_tmp, dim=1)).mean()
                g = torch.autograd.grad(proxy, logits_req, retain_graph=False, create_graph=False)[0]  # [2B,2]
                # convert to diff gradient: d/d(diff) = grad1 - grad0
                adv_grad = (g[:, 1] - g[:, 0])
                adv_grad = adv_grad / (adv_grad.abs().mean().clamp_min(1e-12))  # normalize

            fused_up_aug = self.depth_gater.fuse_from_logits(seq_up, logits_up, perturbed=True, adv_grad=adv_grad)
            aug_user_emb = fused_up_aug[:B]
            aug_item_emb = fused_up_aug[B:]
        else:
            aug_user_emb, aug_item_emb = None, None

        total_loss, _, _ = self.calculate_loss(
            user_emb, pos_emb, neg_emb,
            aug_user_emb=aug_user_emb,
            aug_item_emb=aug_item_emb,
            user_weight=user_weight,
            item_weight=item_weight,
        )
        return total_loss

    @torch.no_grad()
    def forward(self, adj, perturbed=False):
        device = args.device
        N = int(args.user + args.item)
        ids = torch.arange(N, device=device, dtype=torch.long)
        seq = self._build_depth_seqs_for_ids(adj, ids)
        deg = self._get_deg_for_ids(ids)
        out = self.depth_gater(seq, deg=deg, perturbed=perturbed)
        user_embeds = out[: int(args.user)]
        item_embeds = out[int(args.user):]
        return user_embeds, item_embeds

    @torch.no_grad()
    def predict_embeddings(self, adj):
        return self.forward(adj, perturbed=False)

    @torch.no_grad()
    def predict_score(self, user, item, adj):
        self.eval()
        user_embeds, item_embeds = self.forward(adj, perturbed=False)
        user_emb = user_embeds[user]
        item_emb = item_embeds[item]
        score = torch.matmul(user_emb, item_emb.transpose(0, 1))
        return score.cpu().numpy()

    # legacy methods kept
    def train_model(self, data_loader):
        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
        adj = self.sparse_mx_to_torch_sparse_tensor(self.build_adj_matrix(data_loader.adj)).to(args.device)
        self.to(args.device)
        for epoch in range(args.epochs):
            self.train()
            total_loss = 0.0
            for batch_idx, (user, pos, neg) in enumerate(data_loader):
                user = user.to(args.device)
                pos = pos.to(args.device)
                neg = neg.to(args.device)
                optimizer.zero_grad()
                loss = self.calcLosses(user, pos, neg, adj)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if batch_idx % 100 == 0 and batch_idx > 0:
                    print(
                        f'Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}, '
                        f'Total Loss: {total_loss / (batch_idx + 1):.4f}'
                    )
            self.evaluate(getattr(data_loader, "validation_set", None))

    def evaluate(self, validation_set):
        self.eval()
        with torch.no_grad():
            pass

    def predict(self, adj):
        return self.forward(adj, perturbed=False)
