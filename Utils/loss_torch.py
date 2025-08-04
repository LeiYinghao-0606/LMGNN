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



