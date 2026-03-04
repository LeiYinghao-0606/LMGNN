import torch as t
import torch.nn.functional as F
import numpy as np
import json
from typing import Dict, List, Tuple, Union, Any, Optional

# =========================
# your original functions
# =========================

def innerProduct(usrEmbeds, itmEmbeds):
    return t.sum(usrEmbeds * itmEmbeds, dim=-1)

def pairPredict(ancEmbeds, posEmbeds, negEmbeds):
    return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)

def calcRegLoss(model):
    ret = 0
    for W in model.parameters():
        ret += W.norm(2).square()
    # ret += (model.usrStruct + model.itmStruct)
    return ret

def contrastLoss(embeds1, embeds2, nodes, temp):
    embeds1 = F.normalize(embeds1 + 1e-8, p=2)
    embeds2 = F.normalize(embeds2 + 1e-8, p=2)
    pckEmbeds1 = embeds1[nodes]
    pckEmbeds2 = embeds2[nodes]
    nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
    deno = t.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1) + 1e-8
    return -t.log(nume / deno).mean()

# =========================
# NEW: degree bucket utilities
# =========================

PerUserMetric = Union[Dict[int, float], np.ndarray, t.Tensor]
PerUserMetricByK = Dict[int, PerUserMetric]

def _to_numpy_1d(x: Union[np.ndarray, t.Tensor, List[int], List[float]]) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, t.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def build_degree_buckets(
    user_ids: Union[List[int], np.ndarray, t.Tensor],
    user_train_deg: Union[np.ndarray, t.Tensor],
    n_buckets: int = 4,
    mode: str = "quantile",
) -> Tuple[List[List[int]], Dict[str, Any]]:
    """
    Build degree buckets for a given set of users.

    Args:
        user_ids: users involved in evaluation (e.g., testUsers)
        user_train_deg: degree array over ALL users (train-degree only), shape [U]
        n_buckets: number of buckets (default 4)
        mode: 'quantile' supported

    Returns:
        buckets: list of user-id lists, length n_buckets
        meta: dict with bucket thresholds and stats
    """
    u = _to_numpy_1d(user_ids).astype(np.int64)
    deg_all = _to_numpy_1d(user_train_deg).astype(np.float64)
    deg = deg_all[u]

    if mode != "quantile":
        raise ValueError("Only mode='quantile' is supported currently.")

    if n_buckets != 4:
        # You can extend to generic quantile splits; keep 4 by default for paper-friendly Q1~Q4
        qs = np.linspace(0, 1, n_buckets + 1)[1:-1]
        ths = np.quantile(deg, qs).tolist()
        buckets = []
        prev = -np.inf
        for th in ths:
            buckets.append(u[(deg > prev) & (deg <= th)].tolist())
            prev = th
        buckets.append(u[deg > prev].tolist())
        meta = {"mode": mode, "n_buckets": n_buckets, "thresholds": ths}
        return buckets, meta

    # 4-bucket quantiles
    q25, q50, q75 = np.quantile(deg, [0.25, 0.50, 0.75])
    b0 = u[deg <= q25]
    b1 = u[(deg > q25) & (deg <= q50)]
    b2 = u[(deg > q50) & (deg <= q75)]
    b3 = u[deg > q75]
    buckets = [b0.tolist(), b1.tolist(), b2.tolist(), b3.tolist()]
    meta = {"mode": mode, "n_buckets": 4, "thresholds": [float(q25), float(q50), float(q75)]}
    return buckets, meta

def _metric_get_for_users(metric: PerUserMetric, users: List[int]) -> float:
    """
    Aggregate a per-user metric over a subset of users.
    metric can be:
      - dict {u: value}
      - np.ndarray shape [U]
      - torch.Tensor shape [U]
    """
    if len(users) == 0:
        return 0.0

    if isinstance(metric, dict):
        vals = [float(metric[u]) for u in users]
        return float(np.mean(vals))

    arr = _to_numpy_1d(metric).astype(np.float64)
    idx = np.asarray(users, dtype=np.int64)
    return float(arr[idx].mean())

def degree_bucket_report(
    test_users: Union[List[int], np.ndarray, t.Tensor],
    user_train_deg: Union[np.ndarray, t.Tensor],
    per_user_recall_atK: PerUserMetricByK,
    per_user_ndcg_atK: PerUserMetricByK,
    Ks: Tuple[int, ...] = (20, 40),
    bucket_mode: str = "quantile",
    bucket_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Make a structured report for degree-bucket evaluation.

    Inputs expected:
      - per_user_recall_atK[K]: per-user Recall@K (dict or array)
      - per_user_ndcg_atK[K]:   per-user NDCG@K   (dict or array)

    Returns:
      report dict with 'meta' and 'rows'
    """
    if bucket_names is None:
        # paper-friendly default
        if len(Ks) > 0:
            bucket_names = ["Q1 (cold)", "Q2", "Q3", "Q4 (head)"]
        else:
            bucket_names = [f"B{i}" for i in range(4)]

    buckets, meta = build_degree_buckets(test_users, user_train_deg, n_buckets=4, mode=bucket_mode)

    deg_all = _to_numpy_1d(user_train_deg).astype(np.float64)

    rows = []
    for bi, users in enumerate(buckets):
        if len(users) > 0:
            d = deg_all[np.asarray(users, dtype=np.int64)]
            deg_mean = float(d.mean())
            deg_median = float(np.median(d))
            deg_min = float(d.min())
            deg_max = float(d.max())
        else:
            deg_mean = deg_median = deg_min = deg_max = 0.0

        row = {
            "bucket": bucket_names[bi] if bi < len(bucket_names) else f"B{bi}",
            "n_users": int(len(users)),
            "deg_mean": deg_mean,
            "deg_median": deg_median,
            "deg_min": deg_min,
            "deg_max": deg_max,
        }

        for K in Ks:
            if K not in per_user_recall_atK or K not in per_user_ndcg_atK:
                raise KeyError(f"Missing per-user metrics for K={K}.")
            row[f"Recall@{K}"] = _metric_get_for_users(per_user_recall_atK[K], users)
            row[f"NDCG@{K}"] = _metric_get_for_users(per_user_ndcg_atK[K], users)

        rows.append(row)

    report = {"meta": meta, "rows": rows}
    return report

def print_degree_bucket_report(report: Dict[str, Any], Ks: Tuple[int, ...] = (20, 40)):
    """
    Pretty print report returned by degree_bucket_report().
    """
    meta = report.get("meta", {})
    rows = report.get("rows", [])
    ths = meta.get("thresholds", None)

    if ths is not None:
        if len(ths) == 3:
            print(f"\n=== Degree Buckets (quantile) thresholds: q25={ths[0]:.2f}, q50={ths[1]:.2f}, q75={ths[2]:.2f} ===")
        else:
            print(f"\n=== Degree Buckets thresholds: {ths} ===")
    else:
        print("\n=== Degree Buckets Report ===")

    for r in rows:
        s = f"{r['bucket']:>10} | n={r['n_users']:>5d} | deg_mean={r['deg_mean']:.2f} | deg_med={r['deg_median']:.2f}"
        for K in Ks:
            s += f" | R@{K}={r.get(f'Recall@{K}', 0.0):.4f} N@{K}={r.get(f'NDCG@{K}', 0.0):.4f}"
        print(s)

def save_degree_bucket_report_json(report: Dict[str, Any], path: str):
    """
    Save degree bucket report to json for plotting later.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
