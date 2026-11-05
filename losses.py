"""One-direction, grouped multi-positive contrastive loss. Not bidirectional InfoNCE."""
import torch

def multi_positive_loss(logits, positive, legal):
    """L_i = logsumexp(z_ij, j in A_i) - logsumexp(z_ij, j in P_i).

    P_i is a subset of legal A_i. Queries without positives are excluded from
    training only, counted as misses in unconditional evaluation. Return count.
    """
    if logits.shape != positive.shape or logits.shape != legal.shape: raise ValueError('Mask shape mismatch')
    positive, legal = positive.bool(), legal.bool()
    if (positive & ~legal).any(): raise ValueError('Positive outside legal denominator')
    active = positive.any(-1)
    if not active.any(): return logits.sum()*0, 0
    z, p, a = logits[active], positive[active], legal[active]
    value = torch.logsumexp(z.masked_fill(~a, -torch.inf), -1)-torch.logsumexp(z.masked_fill(~p, -torch.inf), -1)
    if not torch.isfinite(value).all(): raise ValueError('Nonfinite contrastive loss')
    return value.mean(), int(active.sum())

def grouped_loss(q, gallery, labels, temperature):
    if temperature <= 0: raise ValueError('Temperature must be positive')
    logits = (gallery @ q)/temperature
    return multi_positive_loss(logits.unsqueeze(0), (labels == 1).unsqueeze(0), (labels >= 0).unsqueeze(0))
