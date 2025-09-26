import os, time, json
import numpy as np
import torch
from torch.optim import AdamW
from torch_geometric.loader import DataLoader as GeoDataLoader

from .config import CFG
from .utils import seed_everything, ensure_dir, json_dump
from .datasets import PolymerGraphDataset
from .model import GNNRegressor
from .metrics import estimate_task_weights, wmae
from .losses import hybrid_loss

def fit_target_scalers_from_df(df, target_cols):
    mu, sigma = {}, {}
    for c in target_cols:
        vals = df[c].dropna().values.astype(np.float32)
        mu[c] = float(vals.mean()) if len(vals)>0 else 0.0
        std = float(vals.std()) if len(vals)>1 else 1.0
        sigma[c] = max(std, 1e-6)
    return mu, sigma

@torch.no_grad()
def validate_one_epoch(model, loader, weights):
    model.eval()
    total = 0.0
    total_cnt = 0
    all_pred, all_true, all_mask = [], [], []
    for batch in loader:
        batch = batch.to(CFG.device)
        pred = model(batch)
        y, mask = batch.y, batch.mask
        loss, mae_k = wmae(pred, y, mask, weights)
        bs = int(batch.num_graphs)
        total += loss.item() * bs
        total_cnt += bs
        all_pred.append(pred.detach().cpu().numpy())
        all_true.append(y.detach().cpu().numpy())
        all_mask.append(mask.detach().cpu().numpy())
    return total/total_cnt, mae_k.detach().cpu().numpy(), (
        np.concatenate(all_pred), np.concatenate(all_true), np.concatenate(all_mask)
    )

def make_datasets_for_fold(train_df, test_df, fold):
    trn_df = train_df[train_df["fold"] != fold].reset_index(drop=True)
    val_df = train_df[train_df["fold"] == fold].reset_index(drop=True)
    dset_trn = PolymerGraphDataset(trn_df, targets=CFG.target_cols, cache_path=os.path.join(CFG.cache_dir, f"train_f{fold}.pt"))
    dset_val = PolymerGraphDataset(val_df, targets=CFG.target_cols, cache_path=os.path.join(CFG.cache_dir, f"val_f{fold}.pt"))
    dset_tst = PolymerGraphDataset(test_df, targets=None,        cache_path=os.path.join(CFG.cache_dir, f"test.pt"))
    train_loader = GeoDataLoader(dset_trn, batch_size=CFG.batch_size, shuffle=True,
                                 num_workers=CFG.num_workers, pin_memory=True, persistent_workers=(CFG.num_workers>0))
    val_loader   = GeoDataLoader(dset_val, batch_size=CFG.batch_size, shuffle=False,
                                 num_workers=CFG.num_workers, pin_memory=True, persistent_workers=(CFG.num_workers>0))
    test_loader  = GeoDataLoader(dset_tst, batch_size=CFG.batch_size, shuffle=False,
                                 num_workers=CFG.num_workers, pin_memory=True, persistent_workers=(CFG.num_workers>0))
    return (trn_df, val_df), (dset_trn, dset_val, dset_tst), (train_loader, val_loader, test_loader)

def train_one_epoch(model, loader, optimizer, scaler, scheduler, epoch, train_mu, train_sigma, train_weights):
    model.train()
    total_loss = 0.0
    total_graphs = 0
    steps_per_epoch = max(1, len(loader))
    for step, batch in enumerate(loader):
        batch = batch.to(CFG.device)
        with torch.cuda.amp.autocast(enabled=CFG.fp16):
            pred = model(batch)
            loss = hybrid_loss(pred, batch.y, batch.mask, train_weights, CFG.aux_lambda, train_mu, train_sigma, CFG.target_cols)
        optimizer.zero_grad(set_to_none=True)
        if CFG.fp16:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if scheduler is not None:
            scheduler.step(epoch - 1 + (step + 1) / steps_per_epoch)
        bs = int(batch.num_graphs)
        total_loss += loss.item() * bs
        total_graphs += bs
    return total_loss / max(1, total_graphs)

def run_fold(train_df, test_df, fold):
    (trn_df, val_df), (dset_trn, dset_val, dset_tst), (train_loader, val_loader, test_loader) = make_datasets_for_fold(train_df, test_df, fold)
    y_val, m_val = dset_val.get_targets_tensor()
    y_val, m_val = y_val.to(CFG.device), m_val.to(CFG.device)
    val_weights, dbg = estimate_task_weights(y_val, m_val, ranges=None)
    mu, sigma = fit_target_scalers_from_df(trn_df, CFG.target_cols)
    y_tr, m_tr = dset_trn.get_targets_tensor()
    y_tr, m_tr = y_tr.to(CFG.device), m_tr.to(CFG.device)
    train_weights, _ = estimate_task_weights(y_tr, m_tr, ranges=None)

    in_node = dset_trn[0].x.shape[1]
    in_edge = dset_trn[0].edge_attr.shape[1]
    model = GNNRegressor(in_node, in_edge, hidden=CFG.hidden, num_layers=CFG.num_layers,
                         n_tasks=len(CFG.target_cols), dropout=CFG.dropout).to(CFG.device)
    optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.fp16)
    try:
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2, eta_min=1e-5)
    except Exception:
        scheduler = None

    best, best_path, meta_path = 1e9, os.path.join(CFG.cache_dir, f"gnn_fold{fold}.pt"), os.path.join(CFG.cache_dir, f"gnn_fold{fold}.json")
    patience = 0
    for epoch in range(1, CFG.num_epochs+1):
        t0=time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, scheduler, epoch, mu, sigma, train_weights)
        val_wmae, val_maeK, (yp, yt, mm) = validate_one_epoch(model, val_loader, val_weights)
        print(f"[F{fold}] E{epoch:02d} | wMAE(val)={val_wmae:.5f} | per-task: " +
              ", ".join([f"{c}={v:.4f}" for c,v in zip(CFG.target_cols, val_maeK)]) +
              f" | time {time.time()-t0:.1f}s")
        if val_wmae < best - 1e-6:
            best = val_wmae; patience = 0
            torch.save(model.state_dict(), best_path)
            meta = {"weights": val_weights.detach().cpu().tolist(), "dbg": dbg}
            json_dump(meta, meta_path)

            import pandas as pd
            oof = pd.DataFrame(yp, columns=[f"pred_{c}" for c in CFG.target_cols])
            oof_true = pd.DataFrame(yt, columns=CFG.target_cols)
            oof_mask = pd.DataFrame(mm, columns=[f"mask_{c}" for c in CFG.target_cols])
            val_ids = val_df.index.values
            for df in (oof, oof_true, oof_mask): df["index"] = val_ids
            oof.to_csv(os.path.join(CFG.cache_dir, f"oof_pred_fold{fold}.csv"), index=False)
            oof_true.to_csv(os.path.join(CFG.cache_dir, f"oof_true_fold{fold}.csv"), index=False)
            oof_mask.to_csv(os.path.join(CFG.cache_dir, f"oof_mask_fold{fold}.csv"), index=False)
        else:
            patience += 1
            if patience >= CFG.es_patience:
                print(f"[F{fold}] Early stopping at epoch {epoch}. Best wMAE={best:.5f}")
                break

    state = torch.load(best_path, map_location=CFG.device)
    model.load_state_dict(state)
    model.eval()
    preds_test = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(CFG.device)
            p = model(batch)
            preds_test.append(p.detach().cpu().numpy())
    preds_test = np.concatenate(preds_test, axis=0).astype(np.float32)
    np.save(os.path.join(CFG.cache_dir, f"pred_test_fold{fold}.npy"), preds_test)
    return best

def kfold_train(train_df, test_df):
    scores=[]
    for f in range(CFG.n_folds):
        print(f"\n=== Fold {f} ===")
        s = run_fold(train_df, test_df, f)
        scores.append(s)
    print("Fold scores:", scores, " | mean:", float(np.mean(scores)))
