# -*- coding: utf-8 -*-
"""Training and cross-validation for Side-Separated Attention MIL."""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from config import SEED
from mil.dataset import SideAwareBagDataset, collate_side_bags, read_bags_by_side
from mil.model import SideSeparatedAttnMIL


N_FOLDS = 5
BATCH_BAGS = 6
MAX_INST_PER_SIDE = 24
EPOCHS = 50
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 6
AMP = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRINT_EVERY = 20


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _autocast_context(device: str, enabled: bool):
    device_type = "cuda" if device == "cuda" else "cpu"
    return torch.amp.autocast(device_type=device_type, enabled=enabled and device == "cuda")


def train_one_epoch(model, loader, optimizer, scaler_amp=None, device: str = DEVICE):
    """Train for one epoch and return mean loss and elapsed seconds."""
    model.train()
    total_loss = 0.0
    n = 0
    t0 = time.time()
    for step, (_, left_list, right_list, valid_mask, ys) in enumerate(loader, 1):
        ys = ys.float().to(device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, scaler_amp is not None):
            logits, _ = model(left_list, right_list, valid_mask)
            loss = F.binary_cross_entropy_with_logits(logits, ys)
        if scaler_amp is not None:
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * ys.size(0)
        n += ys.size(0)
        if step % PRINT_EVERY == 0:
            print(f"  [train] step={step}/{len(loader)} loss={loss.item():.4f}")
    return total_loss / max(n, 1), time.time() - t0


@torch.no_grad()
def eval_model(model, loader, device: str = DEVICE):
    """Evaluate model and return AUC, IDs, labels, and probabilities."""
    model.eval()
    ys_all, ps_all, ids_all = [], [], []
    for pids, left_list, right_list, valid_mask, ys in loader:
        logits, _ = model(left_list, right_list, valid_mask)
        probs = torch.sigmoid(logits).cpu().numpy().tolist()
        ids_all.extend(pids)
        ys_all.extend(ys.numpy().astype(int).tolist())
        ps_all.extend(probs)
    auc = roc_auc_score(ys_all, ps_all) if len(set(ys_all)) == 2 else float("nan")
    return auc, ids_all, ys_all, ps_all


def _safe_overall_auc(oof_df: pd.DataFrame) -> float:
    if oof_df.empty or oof_df["Label"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(oof_df["Label"], oof_df["Prob"]))


def run_cv(
    meta_path: str,
    out_dir: str,
    n_folds: int = N_FOLDS,
    epochs: int = EPOCHS,
    batch_bags: int = BATCH_BAGS,
    max_inst_per_side: int = MAX_INST_PER_SIDE,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    seed: int = SEED,
    device: str = DEVICE,
    eval_every: int = 0,
):
    """
    Run k-fold cross-validation with a fixed epoch count.

    ``eval_every=0`` evaluates validation data only after the final epoch. Set a
    positive value to save validation curves for diagnostics; do not use those
    intermediate values for early stopping unless a nested protocol is added.
    """
    os.makedirs(out_dir, exist_ok=True)
    seed_all(seed)
    bags, labels = read_bags_by_side(meta_path)
    all_ids = sorted(bags.keys())
    y = np.array([labels[pid] for pid in all_ids], dtype=int)

    print(f"Loaded {len(all_ids)} patients")
    print(f"  Positive: {int(y.sum())}, Negative: {int((1 - y).sum())}")
    print(f"  Device: {device}, Folds: {n_folds}, Epochs: {epochs}")
    print(f"  Assumption: patient label is positive if at least one side is positive.")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_summaries = []
    oof_records = []
    log_rows = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(all_ids, y), 1):
        tr_ids = [all_ids[i] for i in tr_idx]
        va_ids = [all_ids[i] for i in va_idx]
        print(f"\n{'=' * 50}")
        print(f"Fold {fold}/{n_folds}  Train: {len(tr_ids)}  Val: {len(va_ids)}")

        ds_tr = SideAwareBagDataset(
            tr_ids, bags, labels, max_inst_per_side=max_inst_per_side, seed=seed + fold
        )
        ds_va = SideAwareBagDataset(
            va_ids,
            bags,
            labels,
            max_inst_per_side=max_inst_per_side,
            seed=seed,
            deterministic_sampling=True,
        )
        dl_tr = DataLoader(
            ds_tr,
            batch_size=batch_bags,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=(device == "cuda"),
            collate_fn=collate_side_bags,
        )
        dl_va = DataLoader(
            ds_va,
            batch_size=batch_bags,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(device == "cuda"),
            collate_fn=collate_side_bags,
        )

        model = SideSeparatedAttnMIL().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scaler_amp = torch.amp.GradScaler("cuda") if (AMP and device == "cuda") else None

        final_auc = float("nan")
        final_ids, final_ys, final_ps = [], [], []
        for epoch in range(1, epochs + 1):
            train_loss, seconds = train_one_epoch(model, dl_tr, optimizer, scaler_amp, device)
            should_eval = epoch == epochs or (eval_every > 0 and epoch % eval_every == 0)
            val_auc = float("nan")
            if should_eval:
                val_auc, final_ids, final_ys, final_ps = eval_model(model, dl_va, device)
                final_auc = val_auc
            log_rows.append(
                {
                    "Fold": fold,
                    "Epoch": epoch,
                    "TrainLoss": train_loss,
                    "Evaluated": should_eval,
                    "ValAUC": val_auc,
                    "Seconds": seconds,
                }
            )
            print(
                f"  [Fold {fold}] epoch={epoch}/{epochs} loss={train_loss:.4f} "
                f"val_auc={val_auc:.4f} time={seconds:.1f}s"
            )

        final_path = os.path.join(out_dir, f"fold{fold}_final.pth")
        torch.save({"model": model.state_dict(), "auc": final_auc, "fold": fold, "epoch": epochs}, final_path)
        fold_summaries.append({"fold": fold, "final_auc": final_auc, "model": final_path})
        for pid, label, prob in zip(final_ids, final_ys, final_ps):
            oof_records.append({"ID": pid, "Label": label, "Prob": prob, "Fold": fold})

    summary_df = pd.DataFrame(fold_summaries)
    oof_df = pd.DataFrame(oof_records)
    log_df = pd.DataFrame(log_rows)
    summary_path = os.path.join(out_dir, "cv_summary_ss_attnmil.csv")
    oof_path = os.path.join(out_dir, "oof_predictions_ss_attnmil.csv")
    log_path = os.path.join(out_dir, "training_log_ss_attnmil.csv")
    summary_df.to_csv(summary_path, index=False)
    oof_df.to_csv(oof_path, index=False)
    log_df.to_csv(log_path, index=False)

    overall_auc = _safe_overall_auc(oof_df)
    print(f"\n{'=' * 50}")
    print("SS-AttnMIL Cross-Validation Complete")
    print(f"  Overall OOF AUC: {overall_auc:.4f}")
    print(f"  Fold AUCs: {[f['final_auc'] for f in fold_summaries]}")
    print(f"  Saved: {summary_path}")
    print(f"  Saved: {oof_path}")
    print(f"  Saved: {log_path}")
    return summary_df, oof_df


def main():
    parser = argparse.ArgumentParser(description="Train SS-AttnMIL with fixed-epoch k-fold CV")
    parser.add_argument("--meta", required=True, help="Path to patch_metadata.csv")
    parser.add_argument("--out", default="results/mil", help="Output directory")
    parser.add_argument("--n_folds", type=int, default=N_FOLDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_bags", type=int, default=BATCH_BAGS)
    parser.add_argument("--max_inst", type=int, default=MAX_INST_PER_SIDE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--eval_every",
        type=int,
        default=0,
        help="Validation interval for diagnostic curves; 0 means final epoch only.",
    )
    args = parser.parse_args()
    run_cv(
        meta_path=args.meta,
        out_dir=args.out,
        n_folds=args.n_folds,
        epochs=args.epochs,
        batch_bags=args.batch_bags,
        max_inst_per_side=args.max_inst,
        lr=args.lr,
        seed=args.seed,
        eval_every=args.eval_every,
    )


if __name__ == "__main__":
    main()
