import torch
import numpy as np

def mae_per_task(y_pred, y_true, mask):
    diff = torch.abs(y_pred - y_true) * mask
    denom = torch.clamp(mask.sum(dim=0), min=1.0)
    mae_k = diff.sum(dim=0) / denom
    return mae_k

def wmae(y_pred, y_true, mask, weights):
    mae_k = mae_per_task(y_pred, y_true, mask)
    return (mae_k * weights).mean(), mae_k

def estimate_task_weights(y_true, mask, ranges=None):
    K = y_true.shape[1]
    n_k = mask.sum(dim=0).cpu().numpy()
    ranges_est = []
    Y = y_true.detach().cpu().numpy()
    M = mask.detach().cpu().numpy()
    for k in range(K):
        vals = Y[M[:,k] > 0.5, k]
        if len(vals) < 2:
            ranges_est.append(1.0)
        else:
            p5, p95 = np.percentile(vals, 5), np.percentile(vals, 95)
            ranges_est.append(max(p95-p5, 1e-6))
    ranges_est = np.array(ranges_est, dtype=np.float32)
    raw = (1.0 / ranges_est) * (1.0 / np.sqrt(n_k + 1e-9))
    raw = raw * (K / (raw.sum() + 1e-12))
    return torch.tensor(raw, dtype=torch.float, device=y_true.device), {"ranges": ranges_est.tolist(), "n": n_k.tolist()}
