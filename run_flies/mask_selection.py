"""
mask_selection.py

Post-processing of SAM2 results {(i, j): (prior_mask, pred_mask, score)}:
coverage filtering and non-maximum suppression to pick the top-K distinct layers.
"""

import numpy as np


def mask_overlap(mask_a, mask_b, metric='iou'):
    """
    Overlap between two boolean masks.
      'iou' : intersection / union
      'ios' : intersection / area of the SMALLER mask. Catches nesting, where
              a thin layer sits entirely inside a big one and plain IoU is low.
    """
    inter = np.logical_and(mask_a, mask_b).sum()
    if metric == 'ios':
        denom = min(mask_a.sum(), mask_b.sum())
    else:
        denom = np.logical_or(mask_a, mask_b).sum()
    return inter / denom if denom > 0 else 0.0


def non_max_suppression_masks(results, top_k=10, metric='ios'):
    """
    Greedy NMS over SAM2 results {(i, j): (prior_mask, pred_mask, score)}.

    Only `top_k` is user-defined. The overlap threshold is chosen
    automatically: the smallest value (strictest suppression) at which greedy
    NMS still returns `top_k` masks. If the pool has fewer than `top_k`
    candidates, everything is returned.

    metric: 'iou' (intersection/union) or 'ios' (intersection/smaller area)
    """
    candidates = sorted(results.items(), key=lambda kv: kv[1][2], reverse=True)
    keys = [k for k, _ in candidates]
    n = len(keys)
    if n <= top_k:
        return {k: results[k] for k in keys}

    bool_masks = [np.asarray(results[k][1]).astype(bool) for k in keys]

    # Precompute symmetric pairwise overlap matrix once
    ov = np.zeros((n, n), dtype=np.float32)
    for a in range(n):
        for b in range(a + 1, n):
            ov[a, b] = ov[b, a] = mask_overlap(bool_masks[a], bool_masks[b], metric)

    def run_nms(thresh):
        kept = []
        for idx in range(n):  # already sorted by score, best first
            if all(ov[idx, k] < thresh for k in kept):
                kept.append(idx)
                if len(kept) >= top_k:
                    break
        return kept

    # Binary search for the smallest threshold that still yields top_k.
    # hi slightly above 1.0 means nothing is ever suppressed, so it always works.
    lo, hi = 0.0, 1.0 + 1e-6
    while hi - lo > 1e-4:
        mid = (lo + hi) / 2
        if len(run_nms(mid)) >= top_k:
            hi = mid
        else:
            lo = mid

    kept = run_nms(hi)
    print(f"NMS auto threshold ({metric}) = {hi:.3f} -> kept {len(kept)} of {n}")
    return {keys[i]: results[keys[i]] for i in kept}


def filter_results_by_coverage(results, min_fraction=0.01, max_fraction=0.5):
    """
    Drop any (i, j) result whose predicted mask covers less than
    `min_fraction` or more than `max_fraction` of the total image pixels --
    masks below min_fraction are usually degenerate/empty-ish predictions,
    and masks above max_fraction are almost always over-segmentations
    rather than a real bounded strip between the lines.
    """
    filtered = {}
    for key, (prior_mask, pred_mask, score) in results.items():
        if pred_mask is None:
            continue
        pred_mask_bool = np.asarray(pred_mask).astype(bool)
        coverage = np.count_nonzero(pred_mask_bool) / pred_mask_bool.size
        if min_fraction <= coverage <= max_fraction:
            filtered[key] = (prior_mask, pred_mask, score)
    return filtered
