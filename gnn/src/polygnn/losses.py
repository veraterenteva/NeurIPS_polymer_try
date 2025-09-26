import torch
from .metrics import wmae

def normalized_mae(y_pred, y_true, mask, mu_dict, sigma_dict, target_cols):
    device = y_pred.device
    mu = torch.tensor([mu_dict[c] for c in target_cols], device=device).view(1,-1)
    sg = torch.tensor([sigma_dict[c] for c in target_cols], device=device).view(1,-1)
    y_true_n = (y_true - mu) / sg
    y_pred_n = (y_pred - mu) / sg
    diff = torch.abs(y_pred_n - y_true_n) * mask
    denom = torch.clamp(mask.sum(dim=0), min=1.0)
    mae_k = diff.sum(dim=0) / denom
    return mae_k.mean(), mae_k

def hybrid_loss(pred, y, mask, train_weights, aux_lambda, mu, sigma, target_cols):
    loss_main, _ = wmae(pred, y, mask, train_weights)
    _, mae_k_norm = normalized_mae(pred, y, mask, mu, sigma, target_cols)
    aux_mask = torch.tensor([0,0,1,1,0], dtype=torch.float32, device=pred.device)
    loss_aux = (mae_k_norm * aux_mask).sum() / aux_mask.sum()
    return loss_main + aux_lambda * loss_aux
